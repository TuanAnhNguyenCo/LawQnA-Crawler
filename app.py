import chainlit as cl 
import requests
from core import *
from crawl import *



@cl.on_chat_start
async def on_chat_start():
    welcome_message = """Chào mừng bạn đến chatbot hỏi luật của chúng tôi. Vui lòng đặt câu hỏi liên quan đến pháp luật Việt Nam."""
    await cl.Message(content=welcome_message).send()

    embedding_model = await load_embedding_model()
    retriever = load_and_create_embeddings(embedding_model = embedding_model)
    llm = await load_llm()

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("embedding_model", embedding_model)
    cl.user_session.set("llm", llm)

@cl.on_message
async def on_message(message: cl.Message):
    user_question = message.content

    # Process the legal question:
    answer = await get_legal_answer(user_question)
     
    # # Send the answer:
    await cl.Message(content=answer).send()


@cl.step(type = 'tool')
def load_temporary_file(user_question):
    crawl_data(user_question)
    embedding_model = cl.user_session.get("embedding_model")
    retriever = load_and_create_embeddings('temporary_data.txt',embedding_model = embedding_model)
    return retriever

@cl.step(type = 'tool')
async def get_legal_answer(user_question):
    llm = cl.user_session.get("llm")
    retriever = cl.user_session.get("retriever")
    answer = await query(user_question,llm,retriever)
    if answer == "Tôi không biết.":
        retriever = load_temporary_file(user_question)
        answer = await query(user_question,llm,retriever,temporary = True)


    return answer

    
