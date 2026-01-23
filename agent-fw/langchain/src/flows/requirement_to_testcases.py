from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from kg.build import build_kg_index, query_kg
from rag.ingest import build_rag_index, load_documents


REQUIREMENT_PROMPT = """You are a senior QA analyst.
Extract and merge requirements from the documents.
Output in Markdown with sections:
1) Functional Requirements
2) Business Rules / Constraints
3) Non-Functional Requirements
4) Open Questions / Missing Info
Keep each item concise and testable.
"""

RISK_PROMPT = """You are a QA engineer focusing on edge cases and risks.
Based on the requirements, list key edge cases, failure scenarios,
and security/abuse cases. Output as a Markdown bullet list.
"""

TESTCASE_PROMPT = """You are a QA engineer.
Based on the merged requirements and risk list, generate functional test cases.
Output in Markdown table with columns:
ID | Title | Preconditions | Steps | Expected Result | Priority

Rules:
- Use clear, testable steps
- Cover positive and negative cases
- Include edge cases from the risk list
- Keep each test case concise
"""

REVIEW_PROMPT = """You are a QA lead.
Review and improve the test cases for completeness, correctness,
and consistency with requirements. Fix missing negatives and
add any critical cases. Output the final Markdown table only.
"""


class FlowState(TypedDict, total=False):
    requirement_summary: str
    risk_list: str
    testcases: str
    final_output: str


class RequirementToTestcaseFlow:
    def __init__(self, config: Dict[str, Any], llm, embeddings):
        self.config = config
        self.llm = llm
        self.embeddings = embeddings

    def run(self) -> str:
        system_cfg = self.config.get("system", {})
        rag_cfg = self.config.get("rag", {})
        kg_cfg = self.config.get("kg", {})

        requirements_dir = system_cfg["requirements_dir"]
        parser_cfg = self.config.get("parser", {})
        documents = load_documents(requirements_dir, parser_cfg)

        rag_index = build_rag_index(
            documents,
            chunk_size=int(rag_cfg.get("chunk_size", 512)),
            chunk_overlap=int(rag_cfg.get("chunk_overlap", 64)),
            splitter_type=str(rag_cfg.get("splitter", "sentence")),
            persist_dir=rag_cfg.get("persist_dir"),
            embeddings=self.embeddings,
        )
        kg_index = build_kg_index(
            documents,
            max_triplets_per_chunk=int(kg_cfg.get("max_triplets_per_chunk", 8)),
            llm=self.llm,
            persist_dir=kg_cfg.get("persist_dir"),
        )

        retriever = rag_index.as_retriever(search_kwargs={"k": 5})

        def build_context(query: str) -> str:
            docs = retriever.invoke(query)
            rag_ctx = "\n\n".join([d.page_content for d in docs])
            kg_ctx = query_kg(kg_index, query)
            return f"## RAG Context\n{rag_ctx}\n\n## KG Context\n{kg_ctx}\n"

        def requirement_node(_state: FlowState) -> FlowState:
            prompt = f"{REQUIREMENT_PROMPT}\n\n{build_context('requirements extraction')}"
            summary = self.llm.invoke(prompt).content
            return {"requirement_summary": summary}

        def risk_node(_state: FlowState) -> FlowState:
            prompt = f"{RISK_PROMPT}\n\n{build_context('risk and edge cases')}"
            risks = self.llm.invoke(prompt).content
            return {"risk_list": risks}

        def testcase_node(state: FlowState) -> FlowState:
            testcase_input = (
                f"{TESTCASE_PROMPT}\n\n"
                f"## Requirement Summary\n{state.get('requirement_summary', '')}\n\n"
                f"## Risk List\n{state.get('risk_list', '')}\n"
            )
            prompt = f"{testcase_input}\n\n{build_context('testcase generation')}"
            testcases = self.llm.invoke(prompt).content
            return {"testcases": testcases}

        def review_node(state: FlowState) -> FlowState:
            review_input = (
                f"{REVIEW_PROMPT}\n\n"
                f"## Requirement Summary\n{state.get('requirement_summary', '')}\n\n"
                f"## Draft Test Cases\n{state.get('testcases', '')}\n"
            )
            prompt = f"{review_input}\n\n{build_context('review testcases')}"
            final_output = self.llm.invoke(prompt).content
            return {"final_output": final_output}

        graph = StateGraph(FlowState)
        graph.add_node("requirement", requirement_node)
        graph.add_node("risk", risk_node)
        graph.add_node("testcase", testcase_node)
        graph.add_node("review", review_node)

        graph.add_edge(START, "requirement")
        graph.add_edge(START, "risk")
        graph.add_edge("requirement", "testcase")
        graph.add_edge("risk", "testcase")
        graph.add_edge("testcase", "review")
        graph.add_edge("review", END)

        app = graph.compile()
        result = app.invoke({})
        return result.get("final_output", "")


def write_output(output_dir: str, content: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "testcases.md"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)

