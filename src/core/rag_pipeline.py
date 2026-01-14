from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LegalRAG:
    def __init__(self):
        # Use a high-quality embedding model (e.g., BAAI/bge-large-en)
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
