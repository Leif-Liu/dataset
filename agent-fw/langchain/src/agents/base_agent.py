from __future__ import annotations

from typing import List

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool


_REACT_TEMPLATE = """You are a helpful assistant that can use tools.

Tools:
{tools}

Use the following format:
Thought: your reasoning
Action: the tool to use, one of [{tool_names}]
Action Input: the input to the tool
Observation: the tool result
... (repeat if needed)
Thought: final reasoning
Final Answer: the final response

User Input: {input}
{agent_scratchpad}
"""


def build_react_agent(llm: BaseChatModel, tools: List[BaseTool]) -> AgentExecutor:
    prompt = PromptTemplate.from_template(_REACT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

