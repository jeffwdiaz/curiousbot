import chromadb
from .data_definitions import CATEGORIES, KNOWLEDGE_BASE

def init_chroma():
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="data/chroma")
    
    # Create or get collections
    try:
        knowledge_base = chroma_client.get_collection("knowledge_base")
        print("Found existing knowledge_base collection")
    except:
        knowledge_base = chroma_client.create_collection("knowledge_base")
        print("Created new knowledge_base collection")
        
    try:
        user_progress = chroma_client.get_collection("user_progress")
        print("Found existing user_progress collection")
    except:
        user_progress = chroma_client.create_collection("user_progress")
        print("Created new user_progress collection")
    
    return chroma_client

if __name__ == "__main__":
    init_chroma() 