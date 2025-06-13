from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

chat_template = ChatPromptTemplate([
    ('system','you are helpfull {domain} expert'),
    ('human','explain me in simple Terms , what is {topic}?')
    # SystemMessage(content='you are helpfull {domain} expert'),
    # HumanMessage(content='explain me in simple Terms , what is {topic}?')
])

prompts = chat_template.invoke({'domain':'cricket','topic':'stumps'})

print(prompts)

