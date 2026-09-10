from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent

load_dotenv()
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")


agent = create_agent(
    model=llm,
    tools=[],
    response_format=ResearchResponse,
    system_prompt="""
    You are a research assistant that will help generate a research paper.

    Answer the user's query and use necessary tools.
    """
)


response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
})

print(response)