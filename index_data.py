import os
import sys
from typing import List, Dict

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sample_data import SAMPLE_DOCUMENTS

class DocumentIndexer:
    """Handles document indexing using LangChain and ChromaDB"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the indexer with configurable chunking parameters
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = "hotel"
        self.storage_dir = "chroma_db"  # Use project root chroma_db
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize text splitter with smart separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        # Initialize embeddings (using the same model as before)
        # Check for CUDA availability
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"\nUsing {device.upper()} for embeddings")
        except ImportError:
            device = "cpu"
            print("\nTorch not found, defaulting to CPU")
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )

    def prepare_documents(self, raw_docs: List[Dict]) -> List[Document]:
        """
        Convert raw documents into LangChain Document objects and split into chunks
        
        Args:
            raw_docs: List of dictionaries containing document content and metadata
            
        Returns:
            List of LangChain Document objects with chunks
        """
        # Convert to LangChain documents
        documents = []
        for idx, doc in enumerate(raw_docs):
            # Add document index to metadata for traceability
            metadata = doc["metadata"].copy()
            metadata.update({
                "doc_id": f"doc_{idx}",
                "source": "sample_data"
            })
            
            langchain_doc = Document(
                page_content=doc["content"].strip(),
                metadata=metadata
            )
            documents.append(langchain_doc)
        
        # Split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        print(f"\nCreated {len(chunked_documents)} chunks from {len(documents)} documents")
        
        # Print chunk statistics
        total_chars = sum(len(doc.page_content) for doc in chunked_documents)
        avg_chunk_size = total_chars / len(chunked_documents)
        print(f"Average chunk size: {avg_chunk_size:.0f} characters")
        
        return chunked_documents

    def index_documents(self, documents: List[Document]) -> Chroma:
        """
        Index documents into ChromaDB using LangChain
        
        Args:
            documents: List of LangChain Document objects to index
            
        Returns:
            Chroma vector store instance
        """
        try:
            print("\nIndexing documents...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.storage_dir,
                collection_name=self.collection_name
            )
            print(f"Successfully indexed {len(documents)} chunks into '{self.collection_name}' collection")
            return vectorstore
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            raise


def main():
    print("Starting document indexing with LangChain...")
    
    try:
        # Initialize indexer with default chunk settings
        indexer = DocumentIndexer()
        
        # Prepare and chunk documents
        documents = indexer.prepare_documents(SAMPLE_DOCUMENTS)
        
        # Print all chunked documents
        print("\nChunked Documents:")
        print("=" * 50)
        for i, doc in enumerate(documents):
            print(f"\nChunk {i + 1}:")
            print("-" * 30)
            print(f"Content:\n{doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 30)
        
        # Index documents
        vectorstore = indexer.index_documents(documents)
        
    except Exception as e:
        print(f"Failed to complete indexing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()