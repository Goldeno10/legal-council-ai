from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LegalRAG:
    """
    Retrieval-Augmented Generation (RAG) pipeline for legal documents.
    Indexes legal texts into a vector database and retrieves relevant clauses
    based on user queries.
    """
    def __init__(self):
        if os.getenv("USE_LOCAL_AI") == "true":
            # Uses Ollama to generate embeddings locally
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        else:
            # Uses high-performance HuggingFace embeddings locally (CPU/GPU)
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.vector_db = None

    def index_document(self, text: str, doc_id: str):
        """
        Splits a long legal doc into chunks and stores them with metadata.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            add_start_index=True
        )
        chunks = splitter.create_documents([text], metadatas=[{"doc_id": doc_id}])
        
        # Store in ChromaDB (local-first for legal privacy)
        self.vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            persist_directory="./data/processed/chroma_db"
        )

    def query_contract(self, query: str):
        """
        Finds the most relevant legal clauses for a specific question.
        """
        if not self.vector_db:
            return "No document indexed."
        
        # Use Similarity Search to find relevant paragraphs
        docs = self.vector_db.similarity_search(query, k=3)
        return docs
