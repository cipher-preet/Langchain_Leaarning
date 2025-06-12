from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "delhi is capital of india",
    "kalkata is capital of west Bengal",
    "paris is capital og france"
]

result = embedding.embed_documents(documents)

print(str(result))
