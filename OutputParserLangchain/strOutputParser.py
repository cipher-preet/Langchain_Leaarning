from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

# from dotenv import load_dotenv
# load_dotenv()
from langchain.prompts import PromptTemplate


llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)


prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

print("this is the result ------>> ", result)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)


print(result1)














# from huggingface_hub import login
# login() # You will be prompted for your HF key, which will then be saved locally
