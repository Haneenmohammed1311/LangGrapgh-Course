from dotenv import load_dotenv
from langchain_core import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

if __name__ == "__main__":
    print("Hello ReAct LangGraph Function Calling")
