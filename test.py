from core import *
load_dotenv()


raw_documents = TextLoader('temporary_data.txt').load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100)

docs = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("Law_db3")
langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="Law_db3",
        embedding_function=embeddings,
)
retriever = langchain_chroma.as_retriever (
        search_type="similarity",
        search_kwargs={"k": 3},
)

print(retriever.invoke("Mũ bảo hiểm"))