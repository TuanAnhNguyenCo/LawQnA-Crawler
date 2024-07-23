from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


def load_and_create_embeddings(file_url = "law.txt",top_k = 3,
                            search_type = 'similarity', embedding_model = None):

    raw_documents = TextLoader(file_url).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100)

    docs = text_splitter.split_documents (raw_documents)

    vector_db = Chroma.from_documents (documents = docs, embedding = embedding_model)

    retriever = vector_db.as_retriever (
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever

async def load_embedding_model(hf_embeddings_model = 'intfloat/multilingual-e5-large-instruct'):
    embedding_model = HuggingFaceEmbeddings(
        model_name=hf_embeddings_model,  
    )
    return embedding_model

async def load_llm(model_name = "gpt-4o-mini-2024-07-18"):
    llm = ChatOpenAI(model=model_name)
    return llm

async def query(USER_QUESTION,llm,retriever,temporary = False):
    template =  """
                Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi.
                Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. 
                Nếu Context tôi cung cấp không có đủ thông tin để trả lời thì hãy trả lời Tôi không biết thay vì cố gắng trả lời sai. 
                Nếu Question không thuộc về luật pháp thì trả lời Không thuộc phạm vi trả lời.
                Trả lời ngắn gọn dễ hiểu tối đa là 3 dòng.
            
                Context: {context}
                Question: {question}
                Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)


    def format_docs (docs) :
        sentences = "\n".join(doc.page_content for doc in docs)
        if temporary:
            with open('law.txt','a+') as f:
                f.write(sentences)
        return sentences

    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough() }
        | prompt
        | llm
    )

   
    output = rag_chain.invoke ( USER_QUESTION )
    return output.content




