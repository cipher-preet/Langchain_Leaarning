import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following - \ {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

# Set a proper user-agent to avoid being blocked by the site
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

url = 'https://www.flipkart.com/apple-macbook-air-m4-16-gb-256-gb-ssd-macos-sequoia-mc6t4hn-a/p/itm7c1831ce25509'

loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'what is the price and cashbaack','text':docs[0].page_content}))