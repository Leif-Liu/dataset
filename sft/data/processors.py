"""
Optimized data processing module for SFT training
优化的SFT训练数据处理模块
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

from datasets import Dataset

from utils.text import ensure_eos, format_prompt


class DataProcessor:
    """优化的数据预处理器"""

    def __init__(
        self,
        tokenizer,
        template: str,
        max_length: int = 2048,
        train_on_prompt: bool = False,
        add_eos_token: bool = True,
        enable_cache: bool = True,
        cache_dir: str = "data/cache",
        num_proc: int = None,
        batch_size: int = 1000,
        enable_validation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.train_on_prompt = train_on_prompt
        self.add_eos_token = add_eos_token
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.num_proc = min(num_proc or 8, os.cpu_count() or 1)
        self.batch_size = batch_size
        self.enable_validation = enable_validation

        # 创建缓存目录
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self) -> str:
        """生成基于配置的缓存键"""
        cache_content = {
            "template": self.template,
            "max_length": self.max_length,
            "train_on_prompt": self.train_on_prompt,
            "add_eos_token": self.add_eos_token,
            "tokenizer_name": getattr(self.tokenizer, 'name_or_path', 'unknown'),
        }
        cache_str = json.dumps(cache_content, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()[:8]

    def get_cache_path(self, dataset_name: str = "train") -> str:
        """获取缓存文件路径"""
        cache_key = self.get_cache_key()
        return os.path.join(self.cache_dir, f"{dataset_name}_{cache_key}.arrow")

    def validate_examples(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """数据验证和清洗"""
        if not self.enable_validation:
            return examples

        cleaned = {"prompt": [], "completion": []}

        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # 1. 基础验证
            if not prompt or not completion:
                continue

            prompt = str(prompt).strip()
            completion = str(completion).strip()

            if len(prompt) < 5 or len(completion) < 5:
                continue

            # 2. 长度检查（粗略估算 4 字符 ≈ 1 token）
            total_chars = len(prompt) + len(completion)
            if total_chars > self.max_length * 4:
                continue

            # 3. 内容清洗
            prompt = prompt.replace('\n\n\n', '\n\n')
            completion = completion.replace('\n\n\n', '\n\n')

            cleaned["prompt"].append(prompt)
            cleaned["completion"].append(completion)

        return cleaned

    def preprocess_batch(self, examples: Dict[str, List[str]]) -> Dict[str, List]:
        """批量优化的预处理函数"""
        batch_size = len(examples["prompt"])

        # 1. 批量构建文本
        full_texts = []
        prompt_texts = []

        for i in range(batch_size):
            prompt_text = format_prompt(self.template, examples["prompt"][i])
            prompt_texts.append(prompt_text)

            full_text = prompt_text + examples["completion"][i]
            if self.add_eos_token:
                full_text = ensure_eos(full_text, self.tokenizer.eos_token)
            full_texts.append(full_text)

        # 2. 批量tokenization（利用tokenizer内部并发）
        full_encodings = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 延迟到collator处理
            add_special_tokens=False,
            return_tensors=None
        )

        prompt_encodings = self.tokenizer(
            prompt_texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
            return_tensors=None
        )

        # 3. 批量构建labels
        results = {"input_ids": [], "attention_mask": [], "labels": []}

        for i in range(batch_size):
            input_ids = full_encodings["input_ids"][i]
            attention_mask = full_encodings["attention_mask"][i]

            labels = input_ids.copy()
            if not self.train_on_prompt:
                prompt_len = len(prompt_encodings["input_ids"][i])
                cut = min(prompt_len, len(labels))
                labels[:cut] = [-100] * cut

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)

        return results

    def process_dataset(
        self,
        dataset: Dataset,
        cache_file_name: str = None,
        desc: str = "Processing SFT data"
    ) -> Dataset:
        """处理数据集主函数"""

        # 1. 数据验证和清洗
        if self.enable_validation:
            print(f"📋 Validating data... (samples: {len(dataset)})")
            dataset = dataset.map(
                self.validate_examples,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                desc="Validating data"
            )
            dataset = dataset.filter(lambda x: len(x["prompt"]) > 0)
            print(f"✅ Validation complete. Valid samples: {len(dataset)}")

        # 2. 确定缓存文件路径
        if cache_file_name is None and self.enable_cache:
            cache_file_name = self.get_cache_path()

        # 3. 批量预处理
        print(f"🚀 Starting batch processing...")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Processes: {self.num_proc}")
        print(f"   - Cache: {'Enabled' if self.enable_cache else 'Disabled'}")
        if cache_file_name:
            print(f"   - Cache file: {cache_file_name}")

        dataset = dataset.map(
            self.preprocess_batch,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=dataset.column_names,
            num_proc=self.num_proc,
            cache_file_name=cache_file_name if self.enable_cache else None,
            desc=desc
        )

        print(f"✅ Processing complete! Processed samples: {len(dataset)}")

        # 4. 显示处理结果统计
        if len(dataset) > 0:
            sample = dataset[0]
            avg_length = sum(len(dataset[i]["input_ids"]) for i in range(min(100, len(dataset)))) / min(100, len(dataset))
            print(f"📊 Dataset statistics:")
            print(f"   - Average sequence length: {avg_length:.1f}")
            print(f"   - Max sequence length: {self.max_length}")
            print(f"   - Sample input_ids shape: {len(sample['input_ids'])}")

        return dataset


class StreamingDataProcessor(DataProcessor):
    """流式数据处理器，用于处理大数据集"""

    def __init__(self, *args, buffer_size: int = 10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size

    def create_streaming_dataset(self, file_path: str) -> Dataset:
        """创建流式数据集"""
        print(f"📡 Creating streaming dataset from {file_path}")
        print(f"   - Buffer size: {self.buffer_size}")

        def data_generator():
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer = []
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        buffer.append(data)
                        if len(buffer) >= self.buffer_size:
                            yield from buffer
                            buffer = []
                    except json.JSONDecodeError:
                        continue

                if buffer:  # 处理剩余数据
                    yield from buffer

        return Dataset.from_generator(data_generator)

    def should_use_streaming(self, file_path: str, threshold: int = 1073741824) -> bool:
        """判断是否应该使用流式处理（默认阈值1GB）"""
        try:
            file_size = os.path.getsize(file_path)
            return file_size > threshold
        except OSError:
            return False


def create_optimized_processor(
    tokenizer,
    cfg,
    enable_cache: bool = True,
    num_proc: int = None,
) -> DataProcessor:
    """创建优化的数据处理器"""

    # 自动检测CPU核数
    if num_proc is None:
        num_proc = min(8, os.cpu_count() or 1)

    # 从配置中获取参数
    template = str(cfg.data.template)
    max_length = int(cfg.tokenizer.max_length)
    train_on_prompt = bool(cfg.sft.train_on_prompt)
    add_eos_token = bool(cfg.sft.add_eos_token)

    # 创建处理器
    processor = DataProcessor(
        tokenizer=tokenizer,
        template=template,
        max_length=max_length,
        train_on_prompt=train_on_prompt,
        add_eos_token=add_eos_token,
        enable_cache=enable_cache,
        cache_dir="data/cache",
        num_proc=num_proc,
        batch_size=1000,
        enable_validation=True,
    )

    return processor