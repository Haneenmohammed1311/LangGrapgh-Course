# The Graph orchestrator
#-----------------------

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2

QUESTION = (
    "Write about AI-Powered SOC / autonomous soc problem domain, "
    "list startups that do that and raised capital."
)


def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Stop once enough search rounds (ToolMessages) have happened."""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()


if __name__ == "__main__":
    # Mermaid source for the graph. Paste it into your README for a rendered diagram.
    print(graph.get_graph().draw_mermaid())

    res = graph.invoke({"messages": [{"role": "user", "content": QUESTION}]})

    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        args = last_message.tool_calls[0]["args"]
        print("\nFINAL ANSWER\n")
        print(args["answer"])
        print("\nREFERENCES")
        for ref in args.get("references", []):
            print("-", ref)