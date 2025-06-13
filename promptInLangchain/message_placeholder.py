from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# chat template

chat_template = ChatPromptTemplate({
    ('system','You are a helpfull customer support agent'),
    MessagesPlaceholder(variable_name='chat_history')
    ('human','{query}')
})


# load chat history
chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readline())


#create Prompt

prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})
