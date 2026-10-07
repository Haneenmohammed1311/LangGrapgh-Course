from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a top-tier LinkedIn thought leader and strict reviewer grading a LinkedIn post. "
            "Generate critical feedback and recommendations for the user's post. "
            "Always provide detailed recommendations, including requests for professional tone, "
            "formatting (bullet points, spacing), hooks, and engagement strategies.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a top-tier LinkedIn content creator. Generate the best possible "
            "LinkedIn post based on the user's request. If the user provides critique, "
            "revise your previous post using that feedback to produce an improved version."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm