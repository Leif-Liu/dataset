from __future__ import annotations

import os
from dataclasses import dataclass
import inspect
import json
import socket
import sys
import tempfile
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from data.loaders import load_sft_dataset
from utils.seed import set_seed
from utils.text import ensure_eos, format_prompt


def _strip_launcher_args(argv: list[str]) -> list[str]:
    """
    DeepSpeed / torchrun 常见会给 user_script 注入参数，例如：
    - --local_rank=0
    - --local_rank 0
    Hydra 默认不认识这些参数，会直接报错。
    """
    cleaned: list[str] = []
    skip_next = False
    for i, a in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if a.startswith("--local_rank=") or a.startswith("--node_rank="):
            continue
        if a in ("--local_rank", "--node_rank"):
            skip_next = True
            continue
        cleaned.append(a)
    return cleaned


# 让 Hydra 看到的 argv 不包含 launcher 注入参数
sys.argv = _strip_launcher_args(sys.argv)


@dataclass
class CausalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _format_bytes(n: int) -> str:
    # show in MiB
    return f"{n / (1024**2):.0f}MiB"


class CudaMemCallback(TrainerCallback):
    """
    轻量级显存观测：在 step 开始重置 peak，在 step 结束打印 allocated/reserved/peak。
    由于 backward/optimizer 的峰值可能发生在 step 内部，所以这里重点看 peak。
    """

    def __init__(self, every_n_steps: int = 1):
        self.every_n_steps = max(1, int(every_n_steps))

    def on_step_begin(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        if state.global_step % self.every_n_steps != 0:
            return
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        if state.global_step % self.every_n_steps != 0:
            return
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        print(
            f"[cuda_mem step={state.global_step}] "
            f"allocated={_format_bytes(alloc)} reserved={_format_bytes(reserved)} peak={_format_bytes(peak)}"
        )


def _dtype_from_cfg(v: str):
    if v in (None, "auto"):
        return "auto"
    m = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return m.get(v, "auto")

def _set_eval_strategy_kwargs(
    evaluation_strategy_value: str,
    eval_steps_value: int | None,
) -> dict[str, Any]:
    """
    Transformers 不同版本参数名不一致：
    - 旧版本：evaluation_strategy
    - 新版本：eval_strategy
    这里做一次运行时兼容。
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = evaluation_strategy_value
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = evaluation_strategy_value
    else:
        # 极端情况：都没有，就不设置
        pass

    if eval_steps_value is not None and "eval_steps" in params:
        kwargs["eval_steps"] = eval_steps_value
    return kwargs


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ensure_single_process_dist_env_for_deepspeed(enabled: bool) -> None:
    """
    当你在单机单进程里直接 `python -m training.sft` 并开启 deepspeed 时，
    DeepSpeed 可能尝试 MPI discovery（进而依赖 mpi4py）。

    这里在“明显是单进程启动”的情况下补齐分布式环境变量，避免走 MPI discovery。
    """
    if not enabled:
        return
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        return
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_pick_free_port()))
    # 即使 DeepSpeed 版本不识别也无害；识别的话可明确禁用 MPI
    os.environ.setdefault("DEEPSPEED_NO_MPI", "1")


def _world_size_from_env() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1


def _materialize_deepspeed_config(
    config_path: str,
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    bf16: bool,
    fp16: bool,
) -> str:
    """
    DeepSpeed config files in this repo use "auto" for batch-size related fields.
    When Transformers enables ZeRO-3 init (deepspeed.zero.Init), DeepSpeed parses the
    config *during* model construction, and older/newer DS versions can choke on
    "auto" strings. Here we write a temp JSON with concrete integers.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    world_size = _world_size_from_env()
    train_micro = int(per_device_train_batch_size)
    gas = int(gradient_accumulation_steps)
    train_batch = int(train_micro * gas * max(1, world_size))

    if cfg.get("train_micro_batch_size_per_gpu") == "auto":
        cfg["train_micro_batch_size_per_gpu"] = train_micro
    if cfg.get("gradient_accumulation_steps") == "auto":
        cfg["gradient_accumulation_steps"] = gas
    if cfg.get("train_batch_size") == "auto":
        cfg["train_batch_size"] = train_batch

    # Keep precision flags consistent with TrainingArguments
    if isinstance(cfg.get("bf16"), dict) and cfg["bf16"].get("enabled") == "auto":
        cfg["bf16"]["enabled"] = bool(bf16)
    if isinstance(cfg.get("fp16"), dict) and cfg["fp16"].get("enabled") == "auto":
        cfg["fp16"]["enabled"] = bool(fp16)

    tmp_dir = tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".deepspeed.json",
        prefix="sft_",
        dir=tmp_dir,
        delete=False,
    ) as wf:
        json.dump(cfg, wf, indent=2, ensure_ascii=False)
        wf.flush()
        return wf.name


def _load_deepspeed_config_dict(path: str) -> dict[str, Any]:
    """
    DeepSpeed accepts json/hjson. Our materialized config is JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            import hjson  # type: ignore

            with open(path, "r", encoding="utf-8") as f:
                return hjson.load(f)
        except Exception:
            return {}


def _init_torch_distributed_if_needed_for_zero_init() -> None:
    """
    HF ZeRO-3 init (deepspeed.zero.Init) constructs DeepSpeedConfig *before* DeepSpeed calls
    init_distributed(). DeepSpeedConfig tries dist.get_world_size()/get_rank(); if torch.distributed
    isn't initialized yet, it falls back to world_size=1 and then batch assertions can fail.
    """
    try:
        import torch.distributed as dist  # type: ignore
    except Exception:
        return

    # Only needed for multi-process launches
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    if dist.is_available() and dist.is_initialized():
        return

    # Ensure ranks/devices are set (DeepSpeed launcher typically exports these)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass

    # Init via env:// (MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE must be present)
    dist.init_process_group(backend="nccl", init_method="env://")


def _init_deepspeed_comm_if_needed_for_zero_init() -> None:
    """
    DeepSpeedConfig (used by ZeRO-3 zero.Init) queries rank/world_size via deepspeed.comm (not
    torch.distributed directly). If deepspeed.comm isn't initialized yet, it falls back to
    world_size=1 and then batch assertions can fail (e.g. 16 != 1*4*1).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    try:
        import deepspeed.comm as ds_comm  # type: ignore
    except Exception:
        return
    try:
        # If already initialized, no-op
        if hasattr(ds_comm, "is_initialized") and ds_comm.is_initialized():
            return
    except Exception:
        pass

    # Ensure CUDA device is set correctly
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass

    # Use env:// (DeepSpeed launcher sets required env vars). Disable MPI discovery explicitly.
    ds_comm.init_distributed(dist_backend="nccl", auto_mpi_discovery=False, init_method="env://")


@hydra.main(version_base=None, config_path="../configs", config_name="sft")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))

    output_dir = to_absolute_path(str(cfg.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # IMPORTANT (ZeRO-3): If you use DeepSpeed ZeRO-3, you must initialize the HF<->DeepSpeed
    # integration *before* calling `from_pretrained()`. Otherwise, each rank may try to materialize
    # the full model on a single GPU during `engine._configure_distributed_model()` and OOM
    # (especially for 30B+ models on 48GB cards).
    #
    # Creating `HfDeepSpeedConfig` early enables parameter partitioning-aware init (zero.Init)
    # where supported by the installed Transformers version.
    deepspeed_config = str(cfg.train.deepspeed_config) if cfg.train.deepspeed_config else None
    if deepspeed_config:
        deepspeed_config = to_absolute_path(deepspeed_config)
        # Must be set before ZeRO-3 init kicks in (it can call deepspeed.comm.init_distributed()).
        _ensure_single_process_dist_env_for_deepspeed(enabled=True)
        # For multi-GPU launches, initialize torch.distributed early so DeepSpeedConfig sees
        # the correct world_size during ZeRO-3 model init.
        _init_torch_distributed_if_needed_for_zero_init()
        # Also initialize DeepSpeed's comm wrapper (DeepSpeedConfig reads world_size from it).
        _init_deepspeed_comm_if_needed_for_zero_init()
        deepspeed_config = _materialize_deepspeed_config(
            deepspeed_config,
            per_device_train_batch_size=int(cfg.train.per_device_train_batch_size),
            gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
            bf16=bool(cfg.train.bf16),
            fp16=bool(cfg.train.fp16),
        )
        try:
            # Transformers >= 4.26
            from transformers.integrations import HfDeepSpeedConfig  # type: ignore
        except Exception:
            try:
                # Older/alternate import path
                from transformers.deepspeed import HfDeepSpeedConfig  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "DeepSpeed config is set, but Transformers DeepSpeed integration is unavailable. "
                    "Please upgrade transformers or install a compatible version."
                ) from e

        # Keep a reference so it is not garbage-collected.
        _dschf = HfDeepSpeedConfig(deepspeed_config)  # noqa: F841

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    tokenizer.padding_side = str(cfg.tokenizer.padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _dtype_from_cfg(str(cfg.model.torch_dtype))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # ===== Optional: LoRA (PEFT) to reduce VRAM/optimizer state =====
    if "peft" in cfg and bool(cfg.peft.enabled):
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PEFT is required for peft.enabled=true. Please `pip install peft`."
            ) from e

        lora_cfg = LoraConfig(
            r=int(cfg.peft.r),
            lora_alpha=int(cfg.peft.lora_alpha),
            lora_dropout=float(cfg.peft.lora_dropout),
            bias=str(cfg.peft.bias),
            task_type="CAUSAL_LM",
            target_modules=list(cfg.peft.target_modules),
        )
        model = get_peft_model(model, lora_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
        # 防御：如果 target_modules 不匹配，可能导致没有任何可训练参数
        trainable = 0
        total = 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        if trainable == 0:
            raise RuntimeError(
                "LoRA/PEFT enabled but no trainable parameters were created. "
                "This usually means `peft.target_modules` does not match the model's module names. "
                "Try overriding e.g. `peft.target_modules='[\"all-linear\"]'` (if your peft version supports it) "
                "or update the target_modules list for Qwen3."
            )

        # 关键：LoRA + gradient checkpointing 时，需要确保输入 embedding 输出 requires_grad=True，
        # 否则 torch.utils.checkpoint 会警告并导致 loss 无法反传梯度。
        if bool(cfg.train.gradient_checkpointing):
            if hasattr(model, "enable_input_require_grads"):
                try:
                    model.enable_input_require_grads()
                except Exception:
                    pass
            else:
                try:
                    emb = model.get_input_embeddings()

                    def _make_output_require_grad(_module, _inp, out):
                        if isinstance(out, torch.Tensor):
                            out.requires_grad_(True)

                    emb.register_forward_hook(_make_output_require_grad)
                except Exception:
                    pass

    if bool(cfg.train.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_file = to_absolute_path(str(cfg.data.train_file))
    ds = load_sft_dataset(
        train_file=train_file,
        prompt_key=str(cfg.data.prompt_key),
        completion_key=str(cfg.data.completion_key),
    )

    template = str(cfg.data.template)
    max_len = int(cfg.tokenizer.max_length)
    train_on_prompt = bool(cfg.sft.train_on_prompt)
    add_eos = bool(cfg.sft.add_eos_token)

    # Optional: ZeRO-3 memory estimate (print-only). Guarded to avoid noise and import cost.
    # Only print on rank0 and only when the configured ZeRO stage is 3.
    # Only for the LoRA part.
    if deepspeed_config and int(os.environ.get("RANK", "0")) == 0:
        ds_cfg = _load_deepspeed_config_dict(deepspeed_config)
        zero_stage = None
        try:
            zero_stage = int(ds_cfg.get("zero_optimization", {}).get("stage", -1))
        except Exception:
            zero_stage = None
        if zero_stage == 3:
            try:
                from deepspeed.runtime.zero.stage3 import (  # type: ignore
                    estimate_zero3_model_states_mem_needs_all_live,
                )

                num_gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
                world_size = int(os.environ.get("WORLD_SIZE", str(num_gpus_per_node)))
                num_nodes = max(1, world_size // max(1, num_gpus_per_node))
                estimate_zero3_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=num_gpus_per_node,
                    num_nodes=num_nodes,
                    additional_buffer_factor=1.5,
                )
            except Exception as e:
                print(f"[warn] ZeRO-3 memory estimate skipped: {e}")

    def preprocess(ex: dict[str, Any]) -> dict[str, Any]:
        prompt = str(ex["prompt"])
        completion = str(ex["completion"])
        prompt_text = format_prompt(template, prompt)
        full_text = prompt_text + completion
        if add_eos:
            full_text = ensure_eos(full_text, tokenizer.eos_token)

        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )["input_ids"]

        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.copy()
        if not train_on_prompt:
            cut = min(len(prompt_ids), len(labels))
            labels[:cut] = [-100] * cut
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    eval_ds = None
    if cfg.data.eval_file:
        eval_file = to_absolute_path(str(cfg.data.eval_file))
        eval_ds = load_sft_dataset(eval_file, str(cfg.data.prompt_key), str(cfg.data.completion_key))
        eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    report_to = "none"
    run_name = None
    if bool(cfg.wandb.enabled):
        report_to = "wandb"
        run_name = str(cfg.wandb.run_name) if cfg.wandb.run_name else None
        os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb.project))

    # (env vars for single-process deepspeed already handled above, before model init)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg.train.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.train.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        warmup_ratio=float(cfg.train.warmup_ratio),
        lr_scheduler_type=str(cfg.train.lr_scheduler_type),
        num_train_epochs=float(cfg.train.num_train_epochs),
        max_steps=int(cfg.train.max_steps),
        logging_steps=int(cfg.train.logging_steps),
        save_steps=int(cfg.train.save_steps),
        bf16=bool(cfg.train.bf16),
        fp16=bool(cfg.train.fp16),
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing),
        deepspeed=deepspeed_config,
        report_to=report_to,
        run_name=run_name,
        **_set_eval_strategy_kwargs(
            evaluation_strategy_value=("steps" if eval_ds is not None and int(cfg.train.eval_steps) > 0 else "no"),
            eval_steps_value=(int(cfg.train.eval_steps) if eval_ds is not None else None),
        ),
        save_total_limit=3,
        logging_first_step=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=CausalLMCollator(tokenizer),
    )

    if bool(getattr(cfg.train, "log_cuda_mem", False)):
        trainer.add_callback(CudaMemCallback(every_n_steps=int(getattr(cfg.train, "cuda_mem_log_steps", 1))))

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


