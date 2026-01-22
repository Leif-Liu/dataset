from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from agents.requirement_agent import build_requirement_agent
from agents.review_agent import build_review_agent
from agents.risk_agent import build_risk_agent
from agents.testcase_agent import build_testcase_agent
from agents.workflow_runner import run_workflow_sync
from kg.build import build_kg_index
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


class RequirementToTestcaseFlow:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self) -> str:
        system_cfg = self.config.get("system", {})
        rag_cfg = self.config.get("rag", {})
        kg_cfg = self.config.get("kg", {})

        requirements_dir = system_cfg["requirements_dir"]
        documents = load_documents(requirements_dir)

        rag_index = build_rag_index(
            documents,
            chunk_size=int(rag_cfg.get("chunk_size", 512)),
            chunk_overlap=int(rag_cfg.get("chunk_overlap", 64)),
        )
        kg_index = build_kg_index(
            documents,
            max_triplets_per_chunk=int(kg_cfg.get("max_triplets_per_chunk", 8)),
        )

        requirement_agent = build_requirement_agent(rag_index, kg_index)
        risk_agent = build_risk_agent(rag_index, kg_index)
        testcase_agent = build_testcase_agent(rag_index, kg_index)
        review_agent = build_review_agent(rag_index, kg_index)

        def run_requirement() -> str:
            return run_workflow_sync(requirement_agent, REQUIREMENT_PROMPT, 6)

        def run_risk() -> str:
            return run_workflow_sync(risk_agent, RISK_PROMPT, 6)

        # Parallel: requirement extraction + risk analysis
        with ThreadPoolExecutor(max_workers=2) as executor:
            req_future = executor.submit(run_requirement)
            risk_future = executor.submit(run_risk)
            requirement_summary = req_future.result()
            risk_list = risk_future.result()

        testcase_input = (
            f"{TESTCASE_PROMPT}\n\n"
            f"## Requirement Summary\n{requirement_summary}\n\n"
            f"## Risk List\n{risk_list}\n"
        )

        testcases = run_workflow_sync(testcase_agent, testcase_input, 6)

        review_input = (
            f"{REVIEW_PROMPT}\n\n"
            f"## Requirement Summary\n{requirement_summary}\n\n"
            f"## Draft Test Cases\n{testcases}\n"
        )
        return run_workflow_sync(review_agent, review_input, 6)


def write_output(output_dir: str, content: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "testcases.md"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)

