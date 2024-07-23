from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
import torch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
import cohere
load_dotenv()


COHERE_API_KEY = os.environ['COHERE_API_KEY']
co = cohere.Client(COHERE_API_KEY)

class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)  # <--- use OpenAIEmbedding's embedding function

    def __call__(self, input):
        return self._embed_documents(input)    # <--- get the embeddings

def load_and_create_embeddings(file_url = "data/law.txt",top_k = 6,
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


def load_llm(model_name = "gpt-4o-mini-2024-07-18"):
    llm = ChatOpenAI(model=model_name)
    return llm

def query(USER_QUESTION,llm,retriever,temporary = False,collection = None,id = None):
    template =  """
                Bạn là trợ lý có trên 20 năm kinh nghiệm cho các nhiệm vụ trả lời câu hỏi dựa trên ngữ cảnh tôi cung cấp.
                Sử dụng các đoạn ngữ cảnh được cung cấp sau đây để trả lời câu hỏi. 
                Nếu ngữ cảnh tôi cung cấp không có đủ thông tin để trả lời thì hãy trả lời Tôi không biết thay vì cố gắng trả lời sai. 
                Trả lời ngắn gọn dễ hiểu không dài dòng lan man.
               
                Ngữ cảnh: {context}
                Câu hỏi: {question}
                Câu trả lời:
    """

    prompt = ChatPromptTemplate.from_template(template)

    def re_rank(docs):
        docs = [
            doc.page_content for doc in docs
        ]

        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query= USER_QUESTION,
            documents=docs,
            top_n=3,
        )
        docs = [docs[i.index] for i in response.results]
        sentences = "\n".join(docs)
        return docs, sentences

    def format_docs (docs) :
        docs, sentences = re_rank(docs)
        if temporary:
            collection.delete(ids = [str(i) for i in range(id[0],id[1]+1)])
            collection.add(
                documents=[doc for doc in docs],
                ids = [str(i+id[0]) for i in range(len(docs))],
                )
            with open('data/law.txt','a+') as f:
                f.write(sentences+"\n\n")
        return sentences

    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough() }
        | prompt
        | llm
    )

   
    output = rag_chain.invoke ( USER_QUESTION )
    return output.content




