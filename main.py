from typing import TypedDict, Annotated
# TypedDict: creates a dictionary type such that a type checker will expect all
            # instances to have a certain set of keys, where each key is
            # associated with a value of a consistent type. This expectation
            # is not checked at runtime.
# Annotated: Add context-specific metadata to a type
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph import add_messages

from chains import generate_chain, reflect_chain 

# typedDict: is a way to define a dictionary type with specific keys and value types.
#  In this case, MessageGraph is a dictionary that has a key "messages"
#  which is associated with a list of BaseMessage objects.
class MessageGraph(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages] 


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state:MessageGraph):
    """
    A node that takes in a MessageGraph and returns a new MessageGraph
    with the generated message from the model.
    """
    return {"messages":[generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    """
    A node that takes in a MessageGraph and returns a new MessageGraph
    with the reflection of the model.

    Before sending to the reflector, swap message roles: the AI's
    drafts become "human" turns (content being submitted for
    critique), and the reflector's own past critiques become "ai"
    turns (its own prior output). This also ensures the final
    message is always a human turn, which Gemini requires.
    """
    cls_map = {"human": AIMessage, "ai": HumanMessage}
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    response = reflect_chain.invoke({"messages": translated})
    return {"messages": [HumanMessage(content=response.content)]}


builder = StateGraph(state_schema = MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

def should_continue(state:MessageGraph):
    """
    A function that takes in a MessageGraph and returns a boolean
    indicating whether the graph should continue or not.
    """

    if len(state["messages"]) > 7:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map ={END: END, REFLECT: REFLECT} )
builder.add_edge(REFLECT, GENERATE)
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
print("Graph drawn to graph.png")

if __name__ == "__main__":
    print("------------- Starting LinkedIn Reflection Agent -------------")
    
    # The real-world prompt
    prompt = """Write a professional LinkedIn post announcing my Graduation Project: 
    A 'Smart Attendance System' built using YOLOv8s-Face, ArcFace, Flask, and Docker, 
    and deployed on Hugging Face Spaces. 
    The goal of the post is to attract AI Engineering recruiters and showcase my MLOps and Computer Vision skills."""
    
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    # Invoke the graph
    res = graph.invoke(inputs)
    
    print("\n" + "="*50)
    print(" LANGGRAPH REFLECTION FLOW ")
    print("="*50 + "\n")
    
    # Loop through all messages to see the critique and revision process
    for msg in res["messages"]:
        msg.pretty_print()