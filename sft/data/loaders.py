from __future__ import annotations

from datasets import load_dataset


def load_jsonl(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    return ds


def load_sft_dataset(
    train_file: str,
    prompt_key: str = "prompt",
    completion_key: str = "completion",
):
    ds = load_jsonl(train_file)
    # normalize columns
    if prompt_key != "prompt" and prompt_key in ds.column_names:
        ds = ds.rename_column(prompt_key, "prompt")
    if completion_key != "completion" and completion_key in ds.column_names:
        ds = ds.rename_column(completion_key, "completion")
    return ds


def load_preference_dataset(
    train_file: str,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
):
    ds = load_jsonl(train_file)
    if prompt_key != "prompt" and prompt_key in ds.column_names:
        ds = ds.rename_column(prompt_key, "prompt")
    if chosen_key != "chosen" and chosen_key in ds.column_names:
        ds = ds.rename_column(chosen_key, "chosen")
    if rejected_key != "rejected" and rejected_key in ds.column_names:
        ds = ds.rename_column(rejected_key, "rejected")
    return ds


