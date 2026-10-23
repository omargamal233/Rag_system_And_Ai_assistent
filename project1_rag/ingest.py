from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_and_split():
    ##load
    loader=TextLoader("D:/GenAi/assinments/ai-projects/project1_rag/data/transformers_notes.txt",encoding="utf-8")
    transformer_doc=loader.load()

    ##split
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    splitted_doc=splitter.split_documents(transformer_doc)
    return splitter.split_documents(splitted_doc)

##embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=r"C:/Users/elber/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
)

##open the saved chroma db (safe to import: this only opens it, it does not change it)
vector_store = Chroma(
    collection_name="project1_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def ingest():
    """Rebuild the collection from scratch. Runs only when this file is executed directly."""
    global vector_store
    vector_store.delete_collection()
    vector_store = Chroma(
        collection_name="project1_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    chunks = load_and_split()
    vector_store.add_documents(chunks)
    print(f"Saved {len(chunks)} chunks to ./chroma_db")


if __name__ == "__main__":
    ingest()