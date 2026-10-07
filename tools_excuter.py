import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearchResults(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    results = []
    for query in search_queries:
        results.append(tavily_tool.invoke({"query": query}))
    return results

execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)