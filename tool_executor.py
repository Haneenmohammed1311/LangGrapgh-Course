# THE TOOLS(the hands)
#---------------------
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import StructuredTool
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    results = []
    for query in search_queries:
        results.append(tavily_tool.invoke({"query": query}))
    return results

# We define a ToolNode that will execute the tools based on the structured output from the LLM. 
# and take the output of the tools and feed it back to the LLM for revision.

execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)