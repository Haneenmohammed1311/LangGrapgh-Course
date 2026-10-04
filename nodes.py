from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import tools, llm

load_dotenv()

SYSTEM_MESSAGE = """You are a helpful assistant that can answer questions 
                    and perform actions using tools."""

# Reasoning Node

def reasoning_node(state: MessagesState) -> MessagesState:
    """
    A node that takes in a MessagesState and returns a new MessagesState
    with the reasoning of the model.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}] + state.messages)
    return {"messages":[response]}

tools_node = ToolNode(tools)
print ("Hello ReAct LangGraph Function Calling")