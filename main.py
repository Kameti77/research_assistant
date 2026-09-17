from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)


agent = create_agent(
    model=llm,
    tools=[search_tool, wiki_tool, save_tool],
    response_format=ResearchResponse,
    system_prompt="""
    You are a research assistant that helps generate research papers.

    Answer the user's research question accurately.

    Use the web search and Wikipedia tools when necessary.

    Use the save tool only once, after you have completed the research
    and generated the final response.

    Do not call the save tool more than once.
    """
)


query = input("What can I help you research? ")

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
)


structured_response = response["structured_response"]

print("\n--- Research Response ---")
print(f"Topic: {structured_response.topic}")
print(f"\nSummary: {structured_response.summary}")
print(f"\nSources: {structured_response.sources}")
print(f"\nTools Used: {structured_response.tools_used}")