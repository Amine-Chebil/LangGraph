from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":
    # Initialize the vector store
    db = Chroma(persist_directory="chroma_db", embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), collection_name="hotel")
    
    # Get all documents (or a sample)
    try:
        collection = db.get()
        docs = collection.get('documents', [])
        metadatas = collection.get('metadatas', [])
        print(f"Total documents in Chroma DB: {len(docs)}")
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            print(f"\nDocument #{i+1}:")
            print(f"Content: {doc}")
            print(f"Metadata: {meta}")
            if i >= 4:
                break
    except Exception as e:
        print(f"Error reading Chroma DB: {e}")
