#!/usr/bin/env python3
"""
数据处理性能基准测试脚本
Performance benchmark script for data processing optimizations
"""
import json
import os
import time
import tempfile
from typing import Dict, List
import argparse

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from data.processors import DataProcessor
from utils.text import format_prompt


def create_sample_data(num_samples: int, avg_length: int = 200) -> List[Dict[str, str]]:
    """创建测试数据"""
    samples = []
    for i in range(num_samples):
        prompt = f"这是第{i+1}个测试问题，长度约{avg_length}字符。" * (avg_length // 20)
        completion = f"这是第{i+1}个测试答案，同样约{avg_length}字符长度。" * (avg_length // 20)
        samples.append({"prompt": prompt, "completion": completion})
    return samples


def benchmark_processing(
    dataset: Dataset,
    tokenizer,
    template: str,
    max_length: int = 2048,
    num_proc_list: List[int] = [1, 2, 4, 8],
    batch_sizes: List[int] = [100, 500, 1000]
) -> Dict:
    """对比不同配置的数据处理性能"""

    results = {}

    print(f"🧪 开始性能基准测试")
    print(f"   - 数据集大小: {len(dataset)} 样本")
    print(f"   - 最大序列长度: {max_length}")
    print(f"   - 测试进程数: {num_proc_list}")
    print(f"   - 测试批次大小: {batch_sizes}")
    print()

    # 1. 测试原始单进程处理（作为baseline）
    print("🔄 测试原始单进程处理...")
    start_time = time.time()

    def simple_preprocess(examples):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            prompt_text = format_prompt(template, prompt)
            full_text = prompt_text + completion

            full_encoding = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )
            prompt_encoding = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )

            input_ids = full_encoding["input_ids"]
            attention_mask = full_encoding["attention_mask"]
            labels = input_ids.copy()

            prompt_len = len(prompt_encoding["input_ids"])
            cut = min(prompt_len, len(labels))
            labels[:cut] = [-100] * cut

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)

        return results

    baseline_ds = dataset.map(simple_preprocess, batched=True, batch_size=100)
    baseline_time = time.time() - start_time

    results["baseline"] = {
        "time": baseline_time,
        "num_proc": 1,
        "batch_size": 100,
        "samples_per_second": len(dataset) / baseline_time
    }
    print(f"   ✅ 基准测试完成: {baseline_time:.2f}s ({results['baseline']['samples_per_second']:.1f} 样本/秒)")
    print()

    # 2. 测试优化后的处理器
    for num_proc in num_proc_list:
        for batch_size in batch_sizes:
            config_name = f"proc{num_proc}_batch{batch_size}"
            print(f"🚀 测试配置: {config_name}")

            try:
                start_time = time.time()

                # 创建临时缓存目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    processor = DataProcessor(
                        tokenizer=tokenizer,
                        template=template,
                        max_length=max_length,
                        train_on_prompt=False,
                        add_eos_token=True,
                        enable_cache=False,  # 为了准确测试，禁用缓存
                        cache_dir=temp_dir,
                        num_proc=num_proc,
                        batch_size=batch_size,
                        enable_validation=True,
                    )

                    # 为每次测试重新创建数据集以确保公平比较
                    test_data = dataset.to_list()
                    test_dataset = Dataset.from_list(test_data)

                    processed_ds = processor.process_dataset(
                        test_dataset,
                        desc=f"Testing {config_name}"
                    )

                process_time = time.time() - start_time
                samples_per_second = len(dataset) / process_time

                results[config_name] = {
                    "time": process_time,
                    "num_proc": num_proc,
                    "batch_size": batch_size,
                    "samples_per_second": samples_per_second,
                    "speedup": samples_per_second / results["baseline"]["samples_per_second"]
                }

                print(f"   ✅ 完成: {process_time:.2f}s ({samples_per_second:.1f} 样本/秒, "
                      f"{results[config_name]['speedup']:.1f}x speedup)")

            except Exception as e:
                print(f"   ❌ 失败: {e}")
                results[config_name] = {"error": str(e)}

            print()

    return results


def print_benchmark_summary(results: Dict):
    """打印基准测试摘要"""
    print("📊 性能基准测试摘要")
    print("=" * 80)

    # 按性能排序
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    sorted_results = sorted(
        valid_results.items(),
        key=lambda x: x[1]["samples_per_second"],
        reverse=True
    )

    print(f"{'配置':<20} {'时间(s)':<10} {'样本/秒':<12} {'加速比':<8} {'进程数':<8} {'批次大小':<8}")
    print("-" * 80)

    for config_name, result in sorted_results:
        speedup = result.get("speedup", 1.0)
        speedup_str = f"{speedup:.1f}x" if speedup != 1.0 else "baseline"

        print(f"{config_name:<20} {result['time']:<10.2f} {result['samples_per_second']:<12.1f} "
              f"{speedup_str:<8} {result['num_proc']:<8} {result['batch_size']:<8}")

    print()

    # 找出最佳配置
    if len(sorted_results) > 1:
        best_config = sorted_results[0]
        print(f"🏆 最佳配置: {best_config[0]}")
        print(f"   - 加速比: {best_config[1]['speedup']:.1f}x")
        print(f"   - 吞吐量: {best_config[1]['samples_per_second']:.1f} 样本/秒")
        print()

    # 错误报告
    error_results = {k: v for k, v in results.items() if "error" in v}
    if error_results:
        print("❌ 失败的配置:")
        for config, result in error_results.items():
            print(f"   - {config}: {result['error']}")


def main():
    parser = argparse.ArgumentParser(description="数据处理性能基准测试")
    parser.add_argument("--samples", type=int, default=1000, help="测试样本数量")
    parser.add_argument("--model", type=str, default="gpt2", help="Tokenizer模型名称")
    parser.add_argument("--max-length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--output", type=str, help="保存结果的JSON文件路径")

    args = parser.parse_args()

    print(f"🔧 初始化tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🔧 生成 {args.samples} 个测试样本...")
    sample_data = create_sample_data(args.samples)
    dataset = Dataset.from_list(sample_data)

    template = "### User:\n{prompt}\n\n### Assistant:\n"

    # 运行基准测试
    results = benchmark_processing(
        dataset=dataset,
        tokenizer=tokenizer,
        template=template,
        max_length=args.max_length,
        num_proc_list=[1, 2, 4, 8],
        batch_sizes=[100, 500, 1000, 2000]
    )

    # 打印摘要
    print_benchmark_summary(results)

    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()