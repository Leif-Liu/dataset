from __future__ import annotations

import os
import random
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.loaders import load_sft_dataset
from utils.seed import set_seed
from utils.text import format_prompt


def _import_ppo():
    # 兼容不同 trl 版本
    try:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer  # type: ignore

        return AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    except Exception:
        from trl import (  # type: ignore
            AutoModelForCausalLMWithValueHead,
            PPOConfig,
            PPOTrainer,
        )

        return AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


def _dtype_from_cfg(v: str):
    if v in (None, "auto"):
        return "auto"
    m = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return m.get(v, "auto")

def _resolve_model_name_or_path(value: str) -> str:
    """
    兼容两种输入：
    - Hugging Face Hub id（例如 Qwen/Qwen2.5-0.5B-Instruct）
    - 本地相对路径（例如 outputs/sft-qwen05b）

    Hydra 会改变 cwd，这里用 to_absolute_path 做一次“存在性判断”，存在则返回绝对路径。
    """
    candidate = to_absolute_path(value)
    return candidate if os.path.exists(candidate) else value


@torch.no_grad()
def score_with_reward_model(
    reward_model,
    reward_tokenizer,
    texts: list[str],
    device: torch.device,
) -> list[torch.Tensor]:
    enc = reward_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    out = reward_model(**enc)
    # logits: [B, 1] -> [B]
    scores = out.logits.squeeze(-1)
    return [s.detach() for s in scores]


@hydra.main(version_base=None, config_path="../configs", config_name="rlhf_ppo")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))

    output_dir = to_absolute_path(str(cfg.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== policy / ref (value head) ======
    AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer = _import_ppo()

    policy_name_or_path = _resolve_model_name_or_path(str(cfg.policy.name_or_path))
    policy_tokenizer = AutoTokenizer.from_pretrained(
        policy_name_or_path,
        trust_remote_code=bool(cfg.policy.trust_remote_code),
    )
    policy_tokenizer.padding_side = str(cfg.tokenizer.padding_side)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    policy_dtype = _dtype_from_cfg(str(cfg.policy.torch_dtype))
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_name_or_path,
        trust_remote_code=bool(cfg.policy.trust_remote_code),
        torch_dtype=policy_dtype,
    )

    # ref_model：用于 KL 约束
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_name_or_path,
        trust_remote_code=bool(cfg.policy.trust_remote_code),
        torch_dtype=policy_dtype,
    )

    # ====== reward model ======
    reward_name_or_path = _resolve_model_name_or_path(str(cfg.reward.name_or_path))
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_name_or_path,
        trust_remote_code=bool(cfg.reward.trust_remote_code),
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_dtype = _dtype_from_cfg(str(cfg.reward.torch_dtype))
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_name_or_path,
        num_labels=1,
        trust_remote_code=bool(cfg.reward.trust_remote_code),
        torch_dtype=reward_dtype,
    ).to(device)
    reward_model.eval()

    # ====== prompts ======
    prompts_file = to_absolute_path(str(cfg.data.prompts_file))
    prompt_key = str(cfg.data.prompt_key)
    template = str(cfg.data.template)

    prompts_ds = load_sft_dataset(prompts_file, prompt_key=prompt_key, completion_key="completion")
    prompts = [format_prompt(template, str(x["prompt"])) for x in prompts_ds]
    if len(prompts) == 0:
        raise ValueError("prompts_file is empty")

    # ====== PPO config ======
    ppo_config = PPOConfig(
        learning_rate=float(cfg.ppo.learning_rate),
        batch_size=int(cfg.ppo.batch_size),
        mini_batch_size=int(cfg.ppo.mini_batch_size),
        gradient_accumulation_steps=int(cfg.ppo.gradient_accumulation_steps),
        kl_coef=float(cfg.ppo.kl_coef),
        log_with=(None if cfg.ppo.log_with in (None, "null") else str(cfg.ppo.log_with)),
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=policy_tokenizer,
    )

    max_prompt_length = int(cfg.tokenizer.max_prompt_length)
    max_new_tokens = int(cfg.tokenizer.max_new_tokens)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(cfg.generation.do_sample),
        "top_p": float(cfg.generation.top_p),
        "temperature": float(cfg.generation.temperature),
        "pad_token_id": policy_tokenizer.pad_token_id,
        "eos_token_id": policy_tokenizer.eos_token_id,
    }

    total_episodes = int(cfg.ppo.total_episodes)
    batch_size = int(cfg.ppo.batch_size)

    for step in range(total_episodes):
        batch_prompts = random.choices(prompts, k=batch_size)
        query_tensors = [
            policy_tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
            for p in batch_prompts
        ]

        response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)

        # 文本用于奖励模型打分
        query_texts = policy_tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        response_texts = policy_tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        rm_texts = [q + r for q, r in zip(query_texts, response_texts)]

        rewards = score_with_reward_model(reward_model, reward_tokenizer, rm_texts, device=device)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        if step % 10 == 0:
            r_mean = torch.stack(rewards).mean().item()
            kl = stats.get("kl", None)
            print(f"[step {step}] reward_mean={r_mean:.4f} kl={kl}")

    # 保存最终策略模型（含 value head）
    ppo_trainer.model.save_pretrained(output_dir)
    policy_tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


