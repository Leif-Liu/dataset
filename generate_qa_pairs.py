#!/usr/bin/env python3
"""
数据增强脚本：将文档内容转换为混合训练数据

功能：
1. 读取现有的instruction_dataset.jsonl（包含文档内容）
2. 为每段文档生成三种类型的训练数据：
   - QA对：问答能力训练
   - 文档续写：细节记忆训练
   - 完整文档：整体记忆训练
3. 支持断点续传、错误重试、批量处理
4. 每个原始文档100%复用，生成多种训练样本
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from datetime import datetime


# ============================================================
# 配置常量
# ============================================================

# API配置（与你的推理环境一致）
API_BASE_URL = "http://10.10.11.7:11541/v1"
API_KEY = "vllm"
MODEL_NAME = "openai-mirror/gpt-oss-120b"

# 生成配置
QUESTIONS_PER_CHUNK = 7      # 每段文档生成的问题数量
MAX_TOKENS = 2048           # 生成回答的最大token数
TEMPERATURE = 0.7            # 生成温度（越高越多样）
BATCH_SIZE = 5              # 每次处理的文档数
REQUEST_DELAY = 10            # 请求间隔（秒）
MAX_RETRIES = 3             # 失败重试次数

# 文档续写配置
CONTINUATION_CHUNK_SIZE = 800    # 文档续写的分段大小（字符数）
CONTINUATION_OVERLAP = 100       # 分段重叠大小

# 文件路径
DEFAULT_INPUT = "processed_md_dataset_Glean_VCU/instruction_dataset.jsonl"
DEFAULT_OUTPUT_DIR = "processed_md_dataset_Glean_VCU/hybrid_dataset"  # 输出目录（包含三个文件）


# ============================================================
# 问题类型模板
# ============================================================

QUESTION_TYPES = [
    "事实问答 - 提取文档中的具体信息",
    "概念解释 - 解释文档中的专业术语或概念",
    "操作步骤 - 描述如何执行某个操作",
    "故障诊断 - 根据文档分析问题原因或解决方案",
    "对比分析 - 比较文档中提到的不同选项或功能",
    "场景应用 - 描述在特定情况下如何使用某功能",
    "参数规格 - 查询具体的技术参数或规格"
]


# ============================================================
# QA生成器
# ============================================================

class QAGenerator:
    """混合训练数据生成器：QA对 + 文档续写 + 完整文档"""

    def __init__(
        self,
        api_base: str = API_BASE_URL,
        api_key: str = API_KEY,
        model: str = MODEL_NAME,
        questions_per_chunk: int = QUESTIONS_PER_CHUNK,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        request_delay: float = REQUEST_DELAY,
        continuation_chunk_size: int = CONTINUATION_CHUNK_SIZE,
        continuation_overlap: int = CONTINUATION_OVERLAP
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.questions_per_chunk = questions_per_chunk
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_delay = request_delay
        self.continuation_chunk_size = continuation_chunk_size
        self.continuation_overlap = continuation_overlap

        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "qa_pairs": 0,
            "continuations": 0,
            "full_docs": 0,
            "total_errors": 0
        }

    def _split_text_for_continuation(self, text: str) -> List[str]:
        """将文本切分为多段，用于生成续写训练数据"""
        if len(text) <= self.continuation_chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.continuation_chunk_size

            # 尝试在句号、换行符等处分割
            if end < len(text):
                for delimiter in ['\n\n', '\n', '。', '. ', '！', '！', '？', '? ']:
                    last_pos = text.rfind(delimiter, start, end)
                    if last_pos != -1:
                        end = last_pos + len(delimiter)
                        break

            chunks.append(text[start:end].strip())
            start = end - self.continuation_overlap

        return [c for c in chunks if len(c) > 50]  # 过滤太短的片段

    def generate_continuation_pairs(self, content: str) -> List[Dict[str, str]]:
        """生成文档续写训练数据"""
        chunks = self._split_text_for_continuation(content)

        if len(chunks) < 2:
            # 文档太短，返回一条完整文档记录
            return [{
                "instruction": "请记住以下文档内容：",
                "input": "",
                "output": content,
                "data_type": "full_document"
            }]

        pairs = []
        for i in range(len(chunks) - 1):
            pairs.append({
                "instruction": "请继续以下内容：",
                "input": chunks[i],
                "output": chunks[i + 1],
                "data_type": "continuation"
            })

        return pairs

    def _create_generation_prompt(self, content: str) -> str:
        """创建QA生成的提示词"""

        prompt = f"""你是一个专业的技术文档分析专家。请根据以下文档内容，生成{self.questions_per_chunk}个高质量的问题-答案对。

文档内容：
```
{content}
```

要求：
1. 问题类型要多样化，包括：
   - 事实问答：提取文档中的具体信息
   - 概念解释：解释文档中的专业术语或概念
   - 操作步骤：描述如何执行某个操作
   - 故障诊断：分析问题原因或解决方案
   - 对比分析：比较不同选项或功能
   - 场景应用：特定情况下的功能使用
   - 参数规格：查询技术参数或规格

2. 问题要具体、有针对性，避免空泛
3. 答案必须基于文档内容，准确完整
4. 问题要用自然语言表达，就像用户在询问一样
5. 对于数据/表格类内容，问题要能引导用户理解数据含义

请以JSON格式输出，格式如下：
```json
{{
  "qa_pairs": [
    {{"question": "...", "answer": "...", "type": "事实问答"}},
    {{"question": "...", "answer": "...", "type": "概念解释"}},
    ...
  ]
}}
```

开始生成："""

        return prompt

    def generate_qa_pairs(self, content: str) -> List[Dict[str, str]]:
        """为一段文档生成QA对（不带文档内容）"""

        prompt = self._create_generation_prompt(content)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的技术文档分析专家，擅长从文档中提取信息并生成高质量的问答对。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content.strip()
                result = json.loads(result_text)

                qa_pairs = result.get("qa_pairs", [])

                if qa_pairs:
                    self.stats["qa_pairs"] += len(qa_pairs)
                    print(f"  ✓ 生成 {len(qa_pairs)} 个QA对")
                    return qa_pairs
                else:
                    print(f"  ⚠ 未生成QA对，重试...")
                    time.sleep(2)
                    continue

            except Exception as e:
                print(f"  ⚠ 请求失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    backoff_time = (attempt + 1) * 5
                    print(f"  等待 {backoff_time} 秒后重试...")
                    time.sleep(backoff_time)
                else:
                    self.stats["total_errors"] += 1
                    return []

        return []

    def generate_full_document(self, content: str, file_name: str) -> Dict[str, str]:
        """生成完整文档训练样本"""

        return {
            "instruction": "请记住以下文档内容，回答问题时需要用到其中的信息：",
            "input": "",
            "output": content,
            "metadata": {
                "data_type": "full_document",
                "source_file": file_name,
                "generated_at": datetime.now().isoformat()
            }
        }

    def process_file(
        self,
        input_file: str,
        output_dir: str,
        resume: bool = True
    ):
        """处理输入文件，生成混合训练数据集"""

        # 检查输入文件
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"❌ 输入文件不存在: {input_file}")
            return

        # 创建输出目录
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        # 定义三个输出文件
        stem = input_path.stem
        qa_file = output_base / f"{stem}_qa.jsonl"
        cont_file = output_base / f"{stem}_continuation.jsonl"
        full_file = output_base / f"{stem}_full.jsonl"

        # 断点续传：检查已处理的文档数量
        processed_count = 0
        if resume:
            # 检查QA文件行数作为处理进度
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            processed_count += 1
                print(f"📂 断点续传：已处理 {processed_count} 个文档")

        # 读取输入文件
        print(f"\n{'='*60}")
        print(f"开始处理: {input_file}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")

        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_index = processed_count

        # 打开三个输出文件
        with open(qa_file, 'a' if resume else 'w', encoding='utf-8') as qa_f, \
             open(cont_file, 'a' if resume else 'w', encoding='utf-8') as cont_f, \
             open(full_file, 'a' if resume else 'w', encoding='utf-8') as full_f:

            for idx in range(start_index, total_lines):
                line = lines[idx].strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    content = data.get('input', '')
                    file_name = data.get('fileName', 'unknown')

                    if not content:
                        continue

                    print(f"[{idx + 1}/{total_lines}] 处理: {file_name}")
                    print(f"  内容长度: {len(content)} 字符")

                    # 1. 生成QA对（不带文档内容）
                    print(f"  生成QA对...")
                    qa_pairs = self.generate_qa_pairs(content)
                    for qa in qa_pairs:
                        qa_sample = {
                            "instruction": qa["question"],
                            "input": "",  # 不带文档内容，让模型记忆知识
                            "output": qa["answer"],
                            "metadata": {
                                "type": qa.get("type", "unknown"),
                                "source_file": file_name,
                                "generated_at": datetime.now().isoformat()
                            }
                        }
                        qa_f.write(json.dumps(qa_sample, ensure_ascii=False) + '\n')

                    # 2. 生成文档续写对
                    print(f"  生成文档续写...")
                    continuation_pairs = self.generate_continuation_pairs(content)
                    for cont in continuation_pairs:
                        if cont["data_type"] == "full_document":
                            # 短文档，保存为完整文档
                            full_sample = {
                                "instruction": "请记住以下文档内容：",
                                "input": "",
                                "output": content,
                                "metadata": {
                                    "data_type": "full_document",
                                    "source_file": file_name,
                                    "generated_at": datetime.now().isoformat()
                                }
                            }
                            full_f.write(json.dumps(full_sample, ensure_ascii=False) + '\n')
                            self.stats["full_docs"] += 1
                        else:
                            # 正常续写对
                            cont_sample = {
                                "instruction": cont["instruction"],
                                "input": cont["input"],
                                "output": cont["output"],
                                "metadata": {
                                    "data_type": "continuation",
                                    "source_file": file_name,
                                    "generated_at": datetime.now().isoformat()
                                }
                            }
                            cont_f.write(json.dumps(cont_sample, ensure_ascii=False) + '\n')
                            self.stats["continuations"] += 1

                    # 3. 同时保存完整文档（确保每个文档都被完整记忆）
                    full_doc_sample = {
                        "instruction": "请记住以下文档内容，回答问题时需要用到其中的信息：",
                        "input": "",
                        "output": content,
                        "metadata": {
                            "data_type": "full_document",
                            "source_file": file_name,
                            "generated_at": datetime.now().isoformat()
                        }
                    }
                    full_f.write(json.dumps(full_doc_sample, ensure_ascii=False) + '\n')
                    self.stats["full_docs"] += 1

                    self.stats["total_processed"] += 1

                    # 请求间隔
                    if idx < total_lines - 1:
                        time.sleep(self.request_delay)

                    # 每5条输出统计
                    if (idx + 1) % 5 == 0:
                        self._print_stats()

                except Exception as e:
                    print(f"  ❌ 处理失败: {e}")
                    self.stats["total_errors"] += 1
                    continue

        # 最终统计
        self._print_final_summary(input_file, output_dir)

    def _print_stats(self):
        """打印中间统计"""
        print(f"\n📊 当前进度:")
        print(f"  已处理文档: {self.stats['total_processed']}")
        print(f"  QA对: {self.stats['qa_pairs']}")
        print(f"  续写对: {self.stats['continuations']}")
        print(f"  完整文档: {self.stats['full_docs']}")
        print(f"  错误数: {self.stats['total_errors']}")
        print()

    def _print_final_summary(self, input_file: str, output_dir: str):
        """打印最终统计"""
        print(f"\n{'='*60}")
        print(f"✅ 处理完成!")
        print(f"  输入文件: {input_file}")
        print(f"  输出目录: {output_dir}")
        print(f"\n📊 数据统计:")
        print(f"  处理文档: {self.stats['total_processed']}")
        print(f"  QA对: {self.stats['qa_pairs']}")
        print(f"  续写对: {self.stats['continuations']}")
        print(f"  完整文档: {self.stats['full_docs']}")
        print(f"  错误数: {self.stats['total_errors']}")
        print(f"\n📁 输出文件:")
        print(f"  - *_qa.jsonl (问答对)")
        print(f"  - *_continuation.jsonl (文档续写)")
        print(f"  - *_full.jsonl (完整文档)")
        print(f"{'='*60}\n")


# ============================================================
# 批量处理多个文件
# ============================================================

def process_multiple_files(
    input_files: List[str],
    output_dir: str,
    generator: QAGenerator
):
    """批量处理多个输入文件"""

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    results = []

    for input_file in input_files:
        print(f"\n{'#'*60}")
        print(f"# 处理文件: {input_file}")
        print(f"{'#'*60}")

        # 重置统计
        generator.stats = {
            "total_processed": 0,
            "qa_pairs": 0,
            "continuations": 0,
            "full_docs": 0,
            "total_errors": 0
        }

        generator.process_file(
            input_file=input_file,
            output_dir=output_dir,
            resume=True
        )

        results.append({
            "input": input_file,
            "stats": generator.stats.copy()
        })

    # 汇总报告
    print(f"\n{'='*60}")
    print(f"📋 全部处理完成汇总")
    print(f"{'='*60}\n")

    total_processed = sum(r["stats"]["total_processed"] for r in results)
    total_qa = sum(r["stats"]["qa_pairs"] for r in results)
    total_cont = sum(r["stats"]["continuations"] for r in results)
    total_full = sum(r["stats"]["full_docs"] for r in results)
    total_errors = sum(r["stats"]["total_errors"] for r in results)

    for r in results:
        print(f"\n文件: {Path(r['input']).name}")
        print(f"  处理文档: {r['stats']['total_processed']}")
        print(f"  QA对: {r['stats']['qa_pairs']}")
        print(f"  续写对: {r['stats']['continuations']}")
        print(f"  完整文档: {r['stats']['full_docs']}")

    print(f"\n{'-'*60}")
    print(f"总计:")
    print(f"  处理文档: {total_processed}")
    print(f"  QA对: {total_qa}")
    print(f"  续写对: {total_cont}")
    print(f"  完整文档: {total_full}")
    print(f"  总训练样本: {total_qa + total_cont + total_full}")
    print(f"  错误数: {total_errors}")
    print(f"{'='*60}\n")

    # 输出合并建议
    print(f"\n💡 后续步骤:")
    print(f"  1. 合并QA对: cat {output_dir}/*_qa.jsonl > {output_dir}/all_qa.jsonl")
    print(f"  2. 合并续写: cat {output_dir}/*_continuation.jsonl > {output_dir}/all_continuation.jsonl")
    print(f"  3. 合并完整文档: cat {output_dir}/*_full.jsonl > {output_dir}/all_full.jsonl")
    print(f"  4. 训练时按比例使用: QA对50% + 续写40% + 完整文档10%\n")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="将文档内容转换为混合训练数据集（QA对 + 文档续写 + 完整文档）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个文件
  python generate_qa_pairs.py --input processed_md_dataset_Glean_VCU/instruction_dataset.jsonl

  # 处理多个文件
  python generate_qa_pairs.py -i file1.jsonl -i file2.jsonl --output-dir sft/data/hybrid

  # 自定义参数
  python generate_qa_pairs.py -i data.jsonl --questions 10 --delay 5

输出文件:
  每个输入文件生成3个输出文件:
  - *_qa.jsonl: QA对（不带文档内容）
  - *_continuation.jsonl: 文档续写对
  - *_full.jsonl: 完整文档
        """
    )

    parser.add_argument(
        "--input", "-i",
        action="append",
        dest="inputs",
        help="输入文件路径（可多次指定）"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）"
    )
    parser.add_argument(
        "--api-base",
        default=API_BASE_URL,
        help=f"API基础URL（默认: {API_BASE_URL})"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"模型名称（默认: {MODEL_NAME})"
    )
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=QUESTIONS_PER_CHUNK,
        help=f"每段文档生成的问题数量（默认: {QUESTIONS_PER_CHUNK}）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"生成回答的最大token数（默认: {MAX_TOKENS})"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=TEMPERATURE,
        help=f"生成温度（默认: {TEMPERATURE})"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=REQUEST_DELAY,
        help=f"请求间隔秒数（默认: {REQUEST_DELAY})"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不启用断点续传（覆盖已有输出）"
    )

    args = parser.parse_args()

    # 默认输入文件
    if not args.inputs:
        args.inputs = [
            "processed_md_dataset/instruction_dataset.jsonl",
            "processed_md_dataset_Glean_VCU/instruction_dataset.jsonl"
        ]
        print(f"⚠ 未指定输入文件，使用默认: {args.inputs}")

    # 创建生成器
    generator = QAGenerator(
        api_base=args.api_base,
        model=args.model,
        questions_per_chunk=args.questions,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_delay=args.delay
    )

    # 处理文件
    process_multiple_files(
        input_files=args.inputs,
        output_dir=args.output_dir,
        generator=generator
    )


if __name__ == "__main__":
    main()
