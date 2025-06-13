from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages = [
    SystemMessage(content='ypu are a heplpfull Assistance'),
    HumanMessage(content='Tell me about python')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)