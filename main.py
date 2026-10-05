from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END

from nodes import  reasoning_node, tools_node

load_dotenv()

REASON_AGENT = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow =  StateGraph(MessagesState)

flow.set_entry_point(REASON_AGENT)
flow.add_node(REASON_AGENT, reasoning_node)
flow.add_node(ACT, tools_node)

flow.add_conditional_edges(REASON_AGENT, should_continue, {
    END:END,
    ACT:ACT})
flow.add_edge(ACT, REASON_AGENT)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    inputs = {"messages": [HumanMessage(content="What is the temperature in Egypt-Mansoura? List it and then triple it")]}
    res = app.invoke(inputs)
    
    print("\n" + "="*50)
    print(" LANGGRAPH STATE FLOW (Messages History)")
    print("="*50 + "\n")
    
    # Loop through all messages in the final state and print them beautifully
    for msg in res["messages"]:
        msg.pretty_print()