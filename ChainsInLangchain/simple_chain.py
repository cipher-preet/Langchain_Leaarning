from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

# simple Chain
chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

# to vusiual the chain
# chain.get_graph().print_ascii()