import chainlit as cl 
import requests
from utils.core import *
from utils.crawler import *



@cl.on_chat_start
async def on_chat_start():
    welcome_message = """Chào mừng bạn đến chatbot hỏi luật của chúng tôi. Vui lòng đặt câu hỏi liên quan đến pháp luật Việt Nam."""
    await cl.Message(content = welcome_message).send()

    retriever, langchain_chroma, collection = load_and_create_embeddings()
    llm = load_llm()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("langchain_chroma", langchain_chroma)
    cl.user_session.set("llm", llm)
    cl.user_session.set("collection", collection)
    
    

@cl.on_message
async def on_message(message: cl.Message):
    user_question = message.content

    # Process the legal question:
    answer = get_legal_answer(user_question)
     
    # # Send the answer:
    await cl.Message(content=answer).send()


@cl.step()
def load_temporary_file(user_question, collection, langchain_chroma):

    crawl_data(user_question,max_page = 3)

    raw_documents = TextLoader('data/temporal_data.txt').load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100)

    docs = text_splitter.split_documents(raw_documents)

    current_id = langchain_chroma._collection.count() + 1

    collection.add(
        documents=[doc.page_content for doc in docs],
        ids = [str(i+current_id) for i in range(len(docs))],
    )

   
    last_id = langchain_chroma._collection.count()

    #remove temporal data file
    os.remove('data/temporal_data.txt')
    return current_id, last_id

   
    
@cl.step()
def get_legal_answer(user_question):
    llm = cl.user_session.get("llm")
    retriever = cl.user_session.get("retriever")
    collection = cl.user_session.get("collection")
    langchain_chroma = cl.user_session.get("langchain_chroma") 

    answer = query(user_question,llm,retriever)
    if answer == "Tôi không biết.":
        current_id, last_id = load_temporary_file(user_question, collection, langchain_chroma)
        answer = query(user_question,llm,retriever,temporary = True,collection = collection,id = [current_id, last_id])

    return answer

    
