from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
"Muttiah Muralitharan Spun magic with wrists, highest Test wickets, unique bowling style.",
"Lasith Malinga Slingy action, deadly yorkers, iconic hair, mastered death overs perfectly.",
"Steve Smith Unorthodox stance, brilliant technique, consistent scorer, quirky but highly effective.",
"Ravindra Jadeja Electric fielder, stylish lefty, accurate spinner, warrior with bat.",
"AB de Villiers Mr. 360, innovator, versatile batsman, destroyed attacks from anywhere."
]


query = "tell me about Muttiah Muralitharan"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)


scores = cosine_similarity([query_embedding],doc_embedding)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(documents[index])
print("simmilarity score is :", score)


