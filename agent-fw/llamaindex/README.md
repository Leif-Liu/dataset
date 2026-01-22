LlamaIndex Agent System - Requirements to Test Cases

Overview
- Root: /home/liufeng/sdk-ragflow/agent-fw/llamaindex
- Purpose: Import and merge user requirement documents and generate functional
  test cases with a LlamaIndex-based agent system.
- Scope: Backend only (no frontend).

Architecture
- Config: YAML-based system configuration.
- Data: Requirement documents (markdown/txt/pdf) under data/requirements.
- Core modules:
  - LLMs: Factory for selecting LLM provider.
  - RAG: Document loading, chunking, and vector indexing.
  - KnowledgeGraph: Graph index for entity/relationship reasoning.
  - Flow: Orchestrates ingestion, indexing, and multi-agent execution.
  - Agent: Multiple tool-augmented agents with RAG + KG query engines.
    - Requirement Agent: extract and merge requirements
    - Risk Agent: edge cases / failure scenarios
    - Testcase Agent: generate test cases
    - Review Agent: validate and refine outputs

Directory Layout
.
├── configs
│   └── system.yaml
├── data
│   └── requirements
│       └── sample_requirements.md
├── outputs
├── src
│   ├── agents
│   │   └── testcase_agent.py
│   ├── flows
│   │   └── requirement_to_testcases.py
│   ├── kg
│   │   └── build.py
│   ├── llms
│   │   └── factory.py
│   ├── rag
│   │   └── ingest.py
│   ├── app.py
│   └── settings.py
└── requirements.txt

Quick Start
1) Install dependencies
   pip install -r /home/liufeng/sdk-ragflow/agent-fw/llamaindex/requirements.txt

2) Run backend
   python /home/liufeng/sdk-ragflow/agent-fw/llamaindex/src/app.py

3) Outputs
   - Console: Generated test cases
   - File: outputs/testcases.md

Notes
- For vLLM (OpenAI-compatible), set:
  - llm.provider: "openai" (or "openai_compatible" for custom model names)
  - llm.base_url: "http://<host>:<port>/v1"
  - llm.model: "<your-model-name>"
  - llm.api_key: "<token>" (or export OPENAI_API_KEY)
  - llm.context_window: 8192 (optional)
- For embeddings with the same OpenAI-compatible server, set:
  - embedding.provider: "openai"
  - embedding.base_url: "http://<host>:<port>/v1"
  - embedding.model: "<your-embedding-model>"
  - embedding.api_key: "<token>" (or export OPENAI_API_KEY)
  - embedding.context_window: 8192 (optional)
- For custom embedding model names (e.g. "BAAI/bge-m3") via OpenAI-compatible
  servers, you can also set embedding.provider to "openai_compatible".
- If you want local embeddings, set embedding.provider to "huggingface"
  and ensure sentence-transformers is installed.

