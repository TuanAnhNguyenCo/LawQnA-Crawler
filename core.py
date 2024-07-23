from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)  # <--- use OpenAIEmbedding's embedding function

    def __call__(self, input):
        return self._embed_documents(input)    # <--- get the embeddings

def load_and_create_embeddings(file_url = "law.txt",top_k = 3,
                            search_type = 'similarity',collection_name = "Law_db"):

    raw_documents = TextLoader(file_url).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100)

    docs = text_splitter.split_documents(raw_documents)

    embeddings = CustomOpenAIEmbeddings(model="text-embedding-3-small")

    persistent_client = chromadb.PersistentClient()

    if collection_name in [c.name for c in persistent_client.list_collections()]:
        collection = persistent_client.get_or_create_collection(collection_name,embedding_function = embeddings)
    else:
        print("Intialize Collection")
        collection = persistent_client.get_or_create_collection(collection_name,embedding_function = embeddings)
        collection.add(
            ids = [str(i+1) for i in range(len(docs))],
            documents=[doc.page_content for doc in docs],
        )
    
    langchain_chroma = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings,
            )
    
    retriever = langchain_chroma.as_retriever (
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever, langchain_chroma, collection


async def load_llm(model_name = "gpt-4o-mini-2024-07-18"):
    llm = ChatOpenAI(model=model_name)
    return llm

async def query(USER_QUESTION,llm,retriever,temporary = False,collection = None,id = None):
    template =  """
                Bạn là trợ lý có trên 20 năm kinh nghiệm cho các nhiệm vụ trả lời câu hỏi dựa trên ngữ cảnh tôi cung cấp.
                Sử dụng các đoạn ngữ cảnh được cung cấp sau đây để trả lời câu hỏi. 
                Nếu ngữ cảnh tôi cung cấp không có đủ thông tin để trả lời thì hãy trả lời Tôi không biết thay vì cố gắng trả lời sai. 
                Trả lời ngắn gọn dễ hiểu tối đa là 3 dòng.
                Chú ý: Nếu câu hỏi không liên quan đến luật pháp thì trả lời Vui lòng hỏi những câu liên quan đến luật.
            
                Ngữ cảnh: {context}
                Câu hỏi: {question}
                Câu trả lời:
    """

    prompt = ChatPromptTemplate.from_template(template)


    def format_docs (docs) :
        sentences = "\n".join(doc.page_content for doc in docs)
        if temporary:
            collection.delete(ids = [str(i) for i in range(id[0],id[1]+1)])
            collection.add(
                documents=[doc.page_content for doc in docs],
                ids = [str(i+id[0]) for i in range(len(docs))],
                )
            with open('law.txt','a+') as f:
                f.write(sentences+"\n\n")
        return sentences

    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough() }
        | prompt
        | llm
    )

   
    output = rag_chain.invoke ( USER_QUESTION )
    return output.content




