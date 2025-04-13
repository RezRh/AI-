import os
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

# For vector similarity search
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not installed. Using simple in-memory vector store instead.")

logger = logging.getLogger("multi_model_assistant.memory")

class MemorySystem:
    """Abstract base class for memory systems."""
    
    def save_interaction(self, query: str, response: Dict[str, Any]):
        """Save an interaction to memory.
        
        Args:
            query: The user's input query
            response: The assistant's response
        """
        pass
    
    def retrieve_relevant(self, query: str, limit: int = 5) -> str:
        """Retrieve relevant past interactions.
        
        Args:
            query: The current query to find relevant memories for
            limit: Maximum number of relevant memories to retrieve
            
        Returns:
            String containing relevant past interactions
        """
        pass

class ChromaMemory(MemorySystem):
    """Memory system using ChromaDB for vector storage and retrieval."""
    
    def __init__(self, collection_name: str = "assistant_memory", 
                 persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB memory system.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required for ChromaMemory but not installed.")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    def save_interaction(self, query: str, response: Dict[str, Any]):
        """Save an interaction to ChromaDB."""
        try:
            # Create a unique ID for this interaction
            interaction_id = f"interaction_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            # Combine query and response for embedding
            text = f"User: {query}\nAssistant: {response['text']}"
            
            # Store in ChromaDB
            self.collection.add(
                documents=[text],
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "response": response["text"],
                    "model": response.get("model", "unknown"),
                    "provider": response.get("provider", "unknown")
                }],
                ids=[interaction_id]
            )
            
            logger.info(f"Saved interaction to memory: {interaction_id}")
        except Exception as e:
            logger.error(f"Error saving interaction to ChromaDB: {e}")
    
    def retrieve_relevant(self, query: str, limit: int = 5) -> str:
        """Retrieve relevant past interactions from ChromaDB."""
        try:
            # Query ChromaDB for similar interactions
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            # Format results as a string
            if results and results["documents"] and results["documents"][0]:
                memories = results["documents"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else []
                
                formatted_memories = []
                for i, memory in enumerate(memories):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    timestamp = metadata.get("timestamp", "Unknown time")
                    formatted_memories.append(f"[{timestamp}] {memory}")
                
                return "\n\n".join(formatted_memories)
            else:
                return ""
        except Exception as e:
            logger.error(f"Error retrieving memories from ChromaDB: {e}")
            return ""

class SimpleMemory(MemorySystem):
    """Simple in-memory vector store for systems without ChromaDB."""
    
    def __init__(self, max_memories: int = 100):
        """Initialize simple memory system.
        
        Args:
            max_memories: Maximum number of memories to store
        """
        self.memories = []
        self.max_memories = max_memories
    
    def save_interaction(self, query: str, response: Dict[str, Any]):
        """Save an interaction to memory."""
        # Create memory entry
        memory = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response["text"],
            "text": f"User: {query}\nAssistant: {response['text']}",
            "embedding": self._simple_embedding(query + " " + response["text"])
        }
        
        # Add to memories, keeping only the most recent up to max_memories
        self.memories.append(memory)
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]
        
        logger.info(f"Saved interaction to simple memory (total: {len(self.memories)})")
    
    def retrieve_relevant(self, query: str, limit: int = 5) -> str:
        """Retrieve relevant past interactions using simple vector similarity."""
        if not self.memories:
            return ""
        
        # Create query embedding
        query_embedding = self._simple_embedding(query)
        
        # Calculate similarities
        similarities = []
        for memory in self.memories:
            similarity = self._cosine_similarity(query_embedding, memory["embedding"])
            similarities.append((similarity, memory))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Format top results
        formatted_memories = []
        for _, memory in similarities[:limit]:
            formatted_memories.append(f"[{memory['timestamp']}] {memory['text']}")
        
        return "\n\n".join(formatted_memories)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create a very simple embedding for text.
        This is a placeholder for a real embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple numpy array embedding
        """
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Create a simple bag-of-words vector (100 dimensions)
        embedding = np.zeros(100)
        
        # Use hash of words to set vector values
        for word in words:
            h = hash(word) % 100
            embedding[h] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)

def create_memory_system(config: Dict) -> MemorySystem:
    """Create a memory system based on configuration.
    
    Args:
        config: Memory configuration dictionary
        
    Returns:
        MemorySystem instance
    """
    provider = config.get("provider", "simple")
    
    if provider == "chroma" and CHROMA_AVAILABLE:
        collection_name = config.get("collection_name", "assistant_memory")
        persist_directory = config.get("persist_directory", "./chroma_db")
        return ChromaMemory(collection_name, persist_directory)
    else:
        if provider == "chroma" and not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available. Falling back to SimpleMemory.")
        return SimpleMemory(max_memories=config.get("max_memories", 100))

# Update the MultiModelAssistant._initialize_memory method to use this factory
def initialize_memory_for_assistant(assistant):
    """Initialize memory system for the assistant.
    
    Args:
        assistant: MultiModelAssistant instance
    """
    try:
        memory_config = assistant.config.get("memory", {"provider": "simple"})
        assistant.memory_system = create_memory_system(memory_config)
        logger.info(f"Initialized memory system: {memory_config['provider']}")
    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        assistant.memory_system = SimpleMemory()
        logger.info("Falling back to SimpleMemory due to initialization error")