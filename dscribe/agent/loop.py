"""The agent loop as a LangGraph state machine.

    START ─► agent ─►(route)─► tools ─► agent ...
                         └────► force_finalize ─► END

`route` enforces the hard step cap, once the step budget is hit,
or the model stops calling tools, control goes to force_finalize, which closes
the draft (auto-marking anything untouched as MISSING) so the output is always
complete and never left mid-flight.
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ..config import CONFIG
from .prompts import SYSTEM_PROMPT, initial_user_message
from ..tools.context import ToolContext
from ..tools.registry import build_tools


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def _page_index_text(ctx: ToolContext) -> str:
    lines = [
        f"  p{p['page_no']:>2}  [{p['doc_type']}/{p['source']}/{p['confidence']}]"
        f"  {p['preview']}"
        for p in ctx.index.list_pages()
    ]
    return "\n".join(lines)


def build_graph(ctx: ToolContext):
    tools = build_tools(ctx)
    llm = ChatOpenAI(
        model=CONFIG.agent_model, temperature=0, timeout=CONFIG.llm_timeout_s
    ).bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        # Carry the model's rationale into the next tool's trace entry.
        content = response.content
        ctx.last_reasoning = content if isinstance(content, str) else json.dumps(content)
        if not getattr(response, "tool_calls", None):
            ctx.trace.log(reasoning=ctx.last_reasoning, action="respond",
                          result="no tool call — concluding")
        return {"messages": [response], "steps": state["steps"] + 1}

    def force_finalize(state: AgentState) -> dict:
        if not ctx.draft.summary.finalized:
            res = ctx.draft.finalize()
            ctx.trace.log(
                reasoning="step cap reached or agent concluded without finalizing"
                if state["steps"] >= CONFIG.max_steps else "agent concluded",
                action="force_finalize", result=json.dumps(res),
            )
        return {}

    def route(state: AgentState) -> str:
        if ctx.draft.summary.finalized:
            return "force_finalize"
        if state["steps"] >= CONFIG.max_steps:
            return "force_finalize"
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "force_finalize"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("force_finalize", force_finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route,
                                {"tools": "tools", "force_finalize": "force_finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("force_finalize", END)
    return graph.compile()


def run_agent(ctx: ToolContext):
    """Run the agent to completion. Returns the finalized DischargeSummary."""
    graph = build_graph(ctx)
    initial = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=initial_user_message(_page_index_text(ctx))),
        ],
        "steps": 0,
    }
    graph.invoke(initial, config={"recursion_limit": CONFIG.max_steps * 2 + 5})
    return ctx.draft.summary
