import json
import os
from pathlib import Path
from openai import OpenAI
import argparse
from typing import List, Dict, Optional

# ============================================================
# 配置常量 - 直接修改以下参数即可
# ============================================================

# OpenAI API 配置
API_BASE_URL = "http://10.10.11.7:11541/v1"
API_KEY = "vllm"
MODEL_NAME = "openai-mirror/gpt-oss-120b"

# Token 配置
MAX_CONTEXT_LENGTH = 131072   # 模型的最大上下文长度
SUMMARY_MAX_TOKENS = 512      # 生成摘要的最大 token 长度

# 文件路径配置
INPUT_DIR = "ragflow_chunks_QA_VCU"
OUTPUT_DIR = "processed_md_dataset"

# 文本分块配置
CHUNK_SIZE = 4000      # 单个分块的最大字符数
CHUNK_OVERLAP = 200    # 分块之间的重叠字符数

# ============================================================


class MDChunkProcessor:
    """处理 Markdown 文件并调用 LLM 生成摘要"""

    def __init__(
        self,
        input_dir: str = INPUT_DIR,
        output_dir: str = OUTPUT_DIR,
        api_base: str = API_BASE_URL,
        api_key: str = API_KEY,
        model: str = MODEL_NAME,
        max_context_length: int = MAX_CONTEXT_LENGTH,
        summary_max_tokens: int = SUMMARY_MAX_TOKENS,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        初始化处理器

        Args:
            input_dir: 输入的 Markdown 文件目录
            output_dir: 输出目录
            api_base: OpenAI API 基础 URL
            api_key: API 密钥
            model: 使用的模型名称
            max_context_length: 模型的最大上下文长度
            summary_max_tokens: 生成摘要的最大 token 数
            chunk_size: 文本分块大小（字符数）
            chunk_overlap: 分块重叠大小
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.max_context_length = max_context_length
        self.summary_max_tokens = summary_max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )

    def read_md_files(self) -> List[Dict[str, str]]:
        """读取目录中的所有 Markdown 文件"""
        md_files = list(self.input_dir.glob("*.md"))
        if not md_files:
            print(f"警告: 在 {self.input_dir} 中未找到 .md 文件")
            return []

        chunks_data = []
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                if content:
                    chunks_data.append({
                        "fileName": md_file.name,
                        "filePath": str(md_file),
                        "content": content
                    })
                    print(f"✓ 读取文件: {md_file.name} ({len(content)} 字符)")
                else:
                    print(f"⚠ 跳过空文件: {md_file.name}")

            except Exception as e:
                print(f"✗ 读取文件失败 {md_file.name}: {e}")

        print(f"\n共读取 {len(chunks_data)} 个 Markdown 文件")
        return chunks_data

    def split_text(self, text: str) -> List[str]:
        """
        将长文本分割成较小的块

        Args:
            text: 输入文本

        Returns:
            分割后的文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # 如果不是最后一块，尝试在句号、换行符等处分割
            if end < len(text):
                # 寻找最近的分割点
                for delimiter in ['\n\n', '\n', '。', '. ', '！', '！', '？', '? ']:
                    last_pos = text.rfind(delimiter, start, end)
                    if last_pos != -1:
                        end = last_pos + len(delimiter)
                        break

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def generate_summary(self, content: str, max_retries: int = 3) -> str:
        """
        调用 LLM 生成摘要

        Args:
            content: 输入内容
            max_retries: 最大重试次数

        Returns:
            生成的摘要
        """
        # 如果内容过长，先进行分块处理
        content_chunks = self.split_text(content)

        if len(content_chunks) == 1:
            return self._generate_summary_single(content_chunks[0], max_retries)
        else:
            # 对每个块生成摘要，然后合并
            chunk_summaries = []
            for i, chunk in enumerate(content_chunks):
                print(f"  处理分块 {i + 1}/{len(content_chunks)}...")
                summary = self._generate_summary_single(chunk, max_retries)
                chunk_summaries.append(summary)

            # 将分块摘要合并后再次生成最终摘要
            combined = "\n\n".join(chunk_summaries)
            return self._generate_summary_single(combined, max_retries)

    def _generate_summary_single(self, content: str, max_retries: int = 3) -> str:
        """生成单个内容的摘要"""
        prompt = f"""请为以下内容生成一个简洁的摘要（3-5句话，突出重点）：

{content}

摘要："""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的内容摘要助手，擅长提取文本的核心要点。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=self.summary_max_tokens
                )
                summary = response.choices[0].message.content.strip()
                return summary

            except Exception as e:
                print(f"  ⚠ API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return f"摘要生成失败: {str(e)}"

    def process_all_files(self):
        """处理所有文件并生成训练数据"""
        print(f"{'='*60}")
        print(f"开始处理 Markdown 文件")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"API 地址: {self.api_base}")
        print(f"模型: {self.model}")
        print(f"{'='*60}\n")

        # 读取所有 MD 文件
        chunks_data = self.read_md_files()
        if not chunks_data:
            print("没有可处理的文件")
            return

        # 为每个文件生成摘要
        print(f"\n开始调用 LLM 生成摘要...\n")
        for idx, chunk in enumerate(chunks_data, 1):
            print(f"[{idx}/{len(chunks_data)}] 处理: {chunk['fileName']}")
            summary = self.generate_summary(chunk['content'])
            chunk['summary'] = summary
            print(f"  摘要: {summary[:100]}...\n")

        # 生成不同格式的训练数据
        self._save_datasets(chunks_data)

    def _save_datasets(self, chunks_data: List[Dict]):
        """保存不同格式的数据集"""

        # 1. 组合格式 (summary + content)
        combined_dataset = []
        for idx, chunk in enumerate(chunks_data):
            combined_text = f"{chunk['summary']}\n\n{chunk['content']}"
            combined_dataset.append({
                "id": idx + 1,
                "text": combined_text,
                "fileName": chunk['fileName']
            })

        combined_path = self.output_dir / "combined_dataset.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_dataset, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存组合数据集: {combined_path} ({len(combined_dataset)} 条)")

        # 2. 问答格式
        qa_dataset = []
        for idx, chunk in enumerate(chunks_data):
            qa_dataset.append({
                "id": idx + 1,
                "question": f"请总结以下内容：\n{chunk['content'][:500]}...",
                "answer": chunk['summary'],
                "fileName": chunk['fileName']
            })

        qa_path = self.output_dir / "qa_dataset.json"
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存问答数据集: {qa_path} ({len(qa_dataset)} 条)")

        # 3. 指令微调格式 (JSONL)
        jsonl_path = self.output_dir / "instruction_dataset.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, chunk in enumerate(chunks_data):
                sample = {
                    "instruction": "请为以下内容生成摘要。",
                    "input": chunk['content'],
                    "output": chunk['summary'],
                    "fileName": chunk['fileName']
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"✓ 已保存指令数据集: {jsonl_path} ({len(chunks_data)} 条)")

        # 4. 原始格式 (包含所有字段)
        raw_path = self.output_dir / "raw_dataset.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存原始数据集: {raw_path} ({len(chunks_data)} 条)")

        # 5. 生成统计信息
        stats = {
            "total_files": len(chunks_data),
            "total_characters": sum(len(c['content']) for c in chunks_data),
            "total_summary_characters": sum(len(c['summary']) for c in chunks_data),
            "average_content_length": sum(len(c['content']) for c in chunks_data) / len(chunks_data),
            "average_summary_length": sum(len(c['summary']) for c in chunks_data) / len(chunks_data),
            "files": [c['fileName'] for c in chunks_data]
        }

        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存统计信息: {stats_path}")

        # 打印统计摘要
        print(f"\n{'='*60}")
        print(f"数据集处理完成！")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  总字符数: {stats['total_characters']:,}")
        print(f"  平均内容长度: {stats['average_content_length']:.0f} 字符")
        print(f"  平均摘要长度: {stats['average_summary_length']:.0f} 字符")
        print(f"{'='*60}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="处理 Markdown 文件并生成训练数据集")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR,
                        help=f"输入的 Markdown 文件目录 (默认: {INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--api-base", type=str, default=API_BASE_URL,
                        help=f"OpenAI API 基础 URL (默认: {API_BASE_URL})")
    parser.add_argument("--api-key", type=str, default=API_KEY,
                        help=f"API 密钥 (默认: {API_KEY})")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"使用的模型名称 (默认: {MODEL_NAME})")
    parser.add_argument("--max-context-length", type=int, default=MAX_CONTEXT_LENGTH,
                        help=f"模型的最大上下文长度 (默认: {MAX_CONTEXT_LENGTH})")
    parser.add_argument("--summary-max-tokens", type=int, default=SUMMARY_MAX_TOKENS,
                        help=f"生成摘要的最大 token 数 (默认: {SUMMARY_MAX_TOKENS})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"文本分块大小 (默认: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP,
                        help=f"分块重叠大小 (默认: {CHUNK_OVERLAP})")

    args = parser.parse_args()

    # 创建处理器并执行
    processor = MDChunkProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_context_length=args.max_context_length,
        summary_max_tokens=args.summary_max_tokens,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    processor.process_all_files()


if __name__ == "__main__":
    main()
