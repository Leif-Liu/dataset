from __future__ import annotations

import os
from dataclasses import dataclass
import inspect
import socket
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

from data.loaders import load_sft_dataset
from utils.seed import set_seed
from utils.text import ensure_eos, format_prompt


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


@hydra.main(version_base=None, config_path="../configs", config_name="sft")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))

    output_dir = to_absolute_path(str(cfg.output_dir))
    os.makedirs(output_dir, exist_ok=True)

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

    deepspeed_config = str(cfg.train.deepspeed_config) if cfg.train.deepspeed_config else None
    if deepspeed_config:
        deepspeed_config = to_absolute_path(deepspeed_config)
    _ensure_single_process_dist_env_for_deepspeed(enabled=bool(deepspeed_config))

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

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


