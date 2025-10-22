#!/usr/bin/env python3
"""
Interview evaluation demo:
- Pre-set interview question
- Collect candidate's spoken answer via Whisper and transcribe to text
- Retrieve RAG context via RAGFlow Agent
- Evaluate the answer using local LLM (gpt-oss-120b) with a strict JSON rubric
- Persist results to JSON

This file orchestrates the flow without modifying existing utilities.
"""

import os
import json
import time
import datetime
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import sounddevice as sd
import whisper
from openai import OpenAI

from ragflow_sdk import RAGFlow, Agent


# ==================== Config ====================
SAMPLE_RATE = 16000
DEFAULT_CAPTURE_SECONDS = 60

# LLM (local deployment)
LLM_MODEL = "openai-mirror/gpt-oss-120b"
LLM_BASE_URL = "http://10.10.11.7:11541/v1"
LLM_API_KEY = "vllm"
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 131072

# Output path
OUTPUT_RESULTS_PATH = "/home/liufeng/sdk-ragflow/chunks_json/interview-results.json"

# Whisper model
WHISPER_MODEL_NAME = "large"

# RAGFlow connection config (standalone; do not import from other files)
RAG_API_KEY = "ragflow-g1ZGRhNjQyNTYzZTExZjA4ZjZiODY2Nj"
RAG_BASE_URL = "http://10.10.11.7:9380"
RAG_AGENT_ID = "a49a6e78ae6611f0a6ff9eec5b87b5d8"


# ==================== Utilities ====================
def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ==================== Whisper Capture ====================
class WhisperCapture:
    def __init__(self, model_name: str = WHISPER_MODEL_NAME):
        print(f"正在加载 Whisper 模型: {model_name} ...")
        self.model = whisper.load_model(model_name)
        print("Whisper 模型加载完成！")

    def capture_answer_once(self, seconds: int = DEFAULT_CAPTURE_SECONDS, sample_rate: int = SAMPLE_RATE) -> Tuple[Optional[str], Optional[str]]:
        """
        Record audio from microphone for a fixed duration and transcribe.
        Returns: (text, language)
        """
        print(f"\n开始录音 {seconds} 秒，采样率 {sample_rate} Hz，按 Ctrl+C 可中断...")
        frames: list[np.ndarray] = []

        def callback(indata, frames_count, time_info, status):
            if status:
                print(f"音频状态: {status}")
            frames.append(indata.copy())

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
                start = time.time()
                while time.time() - start < seconds:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n录音中断，开始转写...")
        except Exception as e:
            print(f"录音错误: {e}")
            return None, None

        if not frames:
            print("未捕获到音频数据")
            return None, None

        audio = np.concatenate(frames).flatten().astype(np.float32)
        try:
            result = self.model.transcribe(
                audio,
                language=None,
                fp16=True,
                verbose=False,
            )
            text = result.get("text", "").strip()
            lang = result.get("language")
            if not text:
                print("未识别到有效文本，请重试录音。")
                return None, lang
            print(f"\n[Whisper识别语言: {lang}] 转写结果: {text}")
            return text, lang
        except Exception as e:
            print(f"转录错误: {e}")
            return None, None


# ==================== RAGFlow Helper ====================
class RAGFlowClient:
    def __init__(self, api_key: str, base_url: str, agent_id: str):
        self.rag_object = RAGFlow(api_key=api_key, base_url=base_url)
        self.agent_id = agent_id
        self.agent: Optional[Agent] = None
        self.session = None
        self._connect()

    def _connect(self) -> None:
        print("正在连接 RAGFlow...")
        print(f"Agent ID: {self.agent_id}")
        try:
            agents_list = self.rag_object.list_agents(id=self.agent_id)
            if not agents_list:
                raise RuntimeError(f"No agent found with ID '{self.agent_id}'")
            self.agent = agents_list[0]
            agent_name = getattr(self.agent, 'name', getattr(self.agent, 'title', 'Unknown'))
            print(f"Using agent: {agent_name} (ID: {self.agent.id})")
            print("正在创建会话...")
            self.session = self.agent.create_session()
            print("Successfully connected to RAGFlow!")
        except Exception as e:
            print(f"连接 RAGFlow 失败: {e}")
            raise

    def fetch_context(self, question: str) -> str:
        """
        Stream the agent's answer for the question and return the final accumulated content
        to be used as RAG context for grading.
        """
        if not self.session:
            raise RuntimeError("RAGFlow session is not initialized")
        cont = ""
        try:
            for ans in self.session.ask(question, stream=True):
                # 兼容不同字段：content / delta / text
                text = getattr(ans, "content", None) or getattr(ans, "delta", None) or getattr(ans, "text", None)
                if text is None:
                    try:
                        text = ans.get("content") or ans.get("delta") or ans.get("text")
                    except Exception:
                        text = None
                if not text:
                    continue
                # 打印增量
                if isinstance(text, str) and cont and text.startswith(cont):
                    print(text[len(cont):], end='', flush=True)
                else:
                    print(text or "", end='', flush=True)
                cont = text if isinstance(text, str) else (cont or "")

            # 流式没有拿到任何内容时，尝试非流式一次
            if not cont:
                resp = self.session.ask(question, stream=False)
                text = None
                if isinstance(resp, str):
                    text = resp
                else:
                    text = getattr(resp, "content", None) or getattr(resp, "delta", None) or getattr(resp, "text", None)
                    if text is None:
                        try:
                            text = resp.get("content") or resp.get("delta") or resp.get("text")
                        except Exception:
                            text = None
                if text:
                    print(text, end='', flush=True)
                    cont = text
        except Exception as e:
            print(f"RAGFlow检索失败: {e}")
            return ""
        return cont or ""


# ==================== Evaluator (gpt-oss-120b) ====================
class Evaluator:
    def __init__(self, base_url: str = LLM_BASE_URL, api_key: str = LLM_API_KEY, model: str = LLM_MODEL):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def build_messages(question: str, rag_context: str, answer: str):
        system_prompt = (
            "You are a strict technical interviewer and grader. Evaluate the candidate's answer ONLY using: "
            "1) The interview question, 2) The RAG context, 3) The candidate's answer transcript. "
            "Do not invent facts. If a claim is not supported by RAG or the answer, treat it as unsupported. "
            "Output JSON ONLY, no markdown or commentary."
        )

        user_prompt = (
            f"[INTERVIEW_QUESTION]\n{question}\n\n"
            f"[RAG_CONTEXT]\n{rag_context}\n\n"
            f"[CANDIDATE_ANSWER]\n{answer}\n\n"
            "[TASK]\n"
            "- Determine how well the answer addresses the question using the RAG context as the ground truth/reference.\n"
            "- Penalize hallucinations (claims not supported by RAG or the question).\n"
            "- Be concise but specific. Cite which RAG points were used.\n\n"
            "[SCORING_RUBRIC]\n"
            "- Knowledge/Understanding: 0-20\n"
            "- Correctness/Accuracy: 0-25\n"
            "- Completeness/Coverage: 0-20\n"
            "- Reasoning/Structure: 0-15\n"
            "- Communication/Clarity: 0-10\n"
            "- Relevance/Focus: 0-10\n"
            "Total: 0-100 (sum of criteria). If information is missing from RAG and the answer relies on it, reduce Correctness and Relevance.\n\n"
            "[OUTPUT_FORMAT]\n"
            "Return a single-line JSON object with:\n"
            "{\n"
            "  \"score_overall\": number,\n"
            "  \"verdict\": \"pass\" | \"marginal\" | \"reject\",\n"
            "  \"criteria\": {\n"
            "    \"knowledge\": number,\n"
            "    \"correctness\": number,\n"
            "    \"completeness\": number,\n"
            "    \"reasoning\": number,\n"
            "    \"communication\": number,\n"
            "    \"relevance\": number\n"
            "  },\n"
            "  \"strengths\": [string],\n"
            "  \"gaps\": [string],\n"
            "  \"suggested_followups\": [string],\n"
            "  \"rag_used_summary\": string,\n"
            "  \"sources\": [ { \"title\": string, \"uri\": string? } ]\n"
            "}\n\n"
            "[VERDICT_THRESHOLDS]\n"
            "- pass: score_overall >= 75\n"
            "- marginal: 60 <= score_overall < 75\n"
            "- reject: score_overall < 60\n\n"
            "[CONSTRAINTS]\n"
            "- Output JSON only. No extra text.\n"
            "- If RAG is empty, set \"rag_used_summary\" to \"No RAG context available\" and apply stronger penalties for correctness and relevance.\n"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def evaluate(self, question: str, rag_context: str, answer: str) -> Dict[str, Any]:
        messages = self.build_messages(question, rag_context, answer)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            content = resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            return {"error": f"LLM调用失败: {e}"}

        # Try to parse a JSON object from model output
        if not content:
            return {"error": "LLM未返回内容"}

        def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(s)
            except Exception:
                # lenient: extract first {...}
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(s[start:end+1])
                    except Exception:
                        return None
                return None

        obj = try_parse_json(content)
        if obj is None:
            return {"error": "LLM输出非JSON", "raw": content}
        return obj


# ==================== Orchestrator ====================
def persist_result(record: Dict[str, Any]) -> None:
    ensure_dir(OUTPUT_RESULTS_PATH)
    try:
        try:
            with open(OUTPUT_RESULTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(record)
        with open(OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[保存] 结果已写入 {OUTPUT_RESULTS_PATH}")
    except Exception as e:
        print(f"[错误] 保存失败: {e}")


def run_demo(question: str, capture_seconds: int = DEFAULT_CAPTURE_SECONDS, answer_text_override: Optional[str] = None):
    print("初始化 RAGFlow 客户端...")
    rag_client = RAGFlowClient(api_key=RAG_API_KEY, base_url=RAG_BASE_URL, agent_id=RAG_AGENT_ID)

    print("准备 Whisper 捕获器...")
    whisper_cap = WhisperCapture()

    # Step 1: RAG retrieval
    t0 = now_str()
    rag_start = time.time()
    print("\n[1/3] 正在检索 RAG 上下文...")
    rag_context = rag_client.fetch_context(question)
    rag_elapsed = f"{time.time() - rag_start:.2f}秒"
    print("RAG 检索内容：")
    print(rag_context)
    print("RAG 检索完成。")

    # Step 2: Capture and transcribe answer
    print("\n[2/3] 请开始回答问题（将进行一次性录音与转写）...")
    if answer_text_override is not None and answer_text_override.strip():
        answer_text = answer_text_override.strip()
        answer_lang = None
        print(f"[使用传入答复文本] {answer_text[:200]}{'...' if len(answer_text)>200 else ''}")
    else:
        answer_text, answer_lang = whisper_cap.capture_answer_once(seconds=capture_seconds)
    if not answer_text:
        print("未获得有效答复，流程结束。")
        return

    # Step 3: Evaluate with LLM
    print("\n[3/3] 正在调用 LLM 进行评分...")
    evaluator = Evaluator()
    llm_start = time.time()
    eval_result = evaluator.evaluate(question, rag_context, answer_text)
    llm_elapsed = f"{time.time() - llm_start:.2f}秒"

    # Record and persist
    record = {
        "timestamp": t0,
        "question": question,
        "rag_context_preview": rag_context[:1000],
        "answer": {
            "text": answer_text,
            "language": answer_lang,
        },
        "evaluation": eval_result,
        "timings": {
            "rag_elapsed": rag_elapsed,
            "llm_elapsed": llm_elapsed,
        },
    }

    persist_result(record)

    # Print a brief summary
    print("\n===== 评分摘要 =====")
    if "error" in eval_result:
        print(f"评分失败: {eval_result['error']}")
    else:
        overall = eval_result.get("score_overall")
        verdict = eval_result.get("verdict")
        print(f"总分: {overall}，结论: {verdict}")
        strengths = eval_result.get("strengths") or []
        gaps = eval_result.get("gaps") or []
        if strengths:
            print("优势:")
            for s in strengths[:5]:
                print(f"- {s}")
        if gaps:
            print("不足:")
            for g in gaps[:5]:
                print(f"- {g}")


def main():
    parser = argparse.ArgumentParser(description="Interview evaluation demo (RAGFlow + Whisper + gpt-oss-120b)")
    parser.add_argument("--question", type=str, default="",
                        help="Interview question; if empty, prompt for input or use default.")
    parser.add_argument("--capture-seconds", type=int, default=DEFAULT_CAPTURE_SECONDS,
                        help="Recording duration for a single answer.")
    parser.add_argument("--answer-text", type=str, default="",
                        help="Skip microphone capture and use this text as the candidate's answer.")
    args = parser.parse_args()

    print("\n===== 面试评估 Demo =====")
    default_question = "Explain how the self-attention mechanism works in Transformers and describe the advantages of multi-head attention."

    question = args.question.strip()
    if not question:
        print("输入问题回车开始；直接回车使用默认示例问题。")
        try:
            user_q = input("问题> ").strip()
        except EOFError:
            user_q = ""
        question = user_q or default_question

    print(f"将使用问题: {question}")
    answer_override = args.answer_text if args.answer_text.strip() else None
    run_demo(question, capture_seconds=args.capture_seconds, answer_text_override=answer_override)


if __name__ == "__main__":
    main()


