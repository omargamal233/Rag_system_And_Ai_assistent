from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import tools
from dotenv import load_dotenv

load_dotenv() 
LLM = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, max_tokens=4096)
tools_list = [
    tools.add,
    tools.sub,
    tools.multiply,
    tools.div,
    tools.time,
    tools.date,
    tools.get_student_grades,
]
agent = create_agent(
    model=LLM,
    tools=tools_list,
    checkpointer=InMemorySaver(),
)