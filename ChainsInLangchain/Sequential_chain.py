from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
) 


prompt2 = PromptTemplate(
    template="Generate a 5 Pointer summary form the following \n {text}",
    input_variables=['text']
) 

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Per capita In India'})

print(result)