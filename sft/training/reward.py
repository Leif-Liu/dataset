from __future__ import annotations

import os
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from data.loaders import load_preference_dataset
from utils.seed import set_seed
from utils.text import ensure_eos, format_prompt


class PairwiseRewardCollator:
    """
    产出 RewardTrainer 常用字段：
    - input_ids_chosen, attention_mask_chosen
    - input_ids_rejected, attention_mask_rejected
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        def pad(seqs, pad_value):
            seqs = [torch.tensor(s, dtype=torch.long) for s in seqs]
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)

        batch = {}
        batch["input_ids_chosen"] = pad([f["input_ids_chosen"] for f in features], self.tokenizer.pad_token_id)
        batch["attention_mask_chosen"] = pad([f["attention_mask_chosen"] for f in features], 0)
        batch["input_ids_rejected"] = pad([f["input_ids_rejected"] for f in features], self.tokenizer.pad_token_id)
        batch["attention_mask_rejected"] = pad([f["attention_mask_rejected"] for f in features], 0)
        return batch


def _dtype_from_cfg(v: str):
    if v in (None, "auto"):
        return "auto"
    m = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return m.get(v, "auto")


def _import_reward_trainer():
    # TRL 不同版本 import 路径可能略有差异，这里做兼容
    try:
        from trl import RewardTrainer  # type: ignore

        return RewardTrainer
    except Exception:
        from trl.trainer import RewardTrainer  # type: ignore

        return RewardTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="reward")
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
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=1,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        torch_dtype=torch_dtype,
    )

    train_file = to_absolute_path(str(cfg.data.train_file))
    ds = load_preference_dataset(
        train_file=train_file,
        prompt_key=str(cfg.data.prompt_key),
        chosen_key=str(cfg.data.chosen_key),
        rejected_key=str(cfg.data.rejected_key),
    )

    template = str(cfg.data.template)
    max_len = int(cfg.tokenizer.max_length)

    def preprocess(ex: dict[str, Any]) -> dict[str, Any]:
        prompt = str(ex["prompt"])
        chosen = str(ex["chosen"])
        rejected = str(ex["rejected"])
        prompt_text = format_prompt(template, prompt)

        chosen_text = ensure_eos(prompt_text + chosen, tokenizer.eos_token)
        rejected_text = ensure_eos(prompt_text + rejected, tokenizer.eos_token)

        chosen_enc = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )
        rejected_enc = tokenizer(
            rejected_text,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )
        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    eval_ds = None

    report_to = "none"
    run_name = None
    if bool(cfg.wandb.enabled):
        report_to = "wandb"
        run_name = str(cfg.wandb.run_name) if cfg.wandb.run_name else None
        os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb.project))

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
        report_to=report_to,
        run_name=run_name,
        evaluation_strategy="no",
        save_total_limit=3,
        logging_first_step=True,
        remove_unused_columns=False,
    )

    RewardTrainer = _import_reward_trainer()
    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=PairwiseRewardCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


