"""Shared memory store management for all tools."""

import logging
from typing import Dict, List

try:
    from langgraph.store.memory import InMemoryStore
except ImportError:
    InMemoryStore = None

logger = logging.getLogger(__name__)


class MemoryStoreManager:
    """
    Singleton manager for LangMem stores per conversation.

    CRITICAL: All tools must use this shared manager to ensure
    memory updates are visible across tools in the same episode.
    """

    _stores: Dict[str, any] = {}  # namespace -> store
    _memory_cache: Dict[str, List[Dict]] = {}  # namespace -> current memories
    _instance_to_namespace: Dict[str, str] = {}  # instance_id -> intended namespace mapping

    @classmethod
    def get_or_create_store(cls, namespace: str) -> any:
        """Get existing store or create new one for namespace."""
        if namespace not in cls._stores:
            if InMemoryStore is None:
                logger.warning(
                    f"LangMem not available, using mock store for {namespace}"
                )
                cls._stores[namespace] = MockMemoryStore(namespace)
            else:
                # Create simple text-based store without embeddings
                cls._stores[namespace] = InMemoryStore()
                print(f"✅ Created LangMem store for namespace: {namespace}")

        return cls._stores[namespace]

    @classmethod
    def init_store_with_memories(cls, namespace: str, initial_memories: List[Dict]):
        """Initialize a store with pre-existing memories."""
        store = cls.get_or_create_store(namespace)

        # Cache the memories for reward computation
        cls._memory_cache[namespace] = initial_memories.copy()

        # 🔧 DEFERRED: Store initial population will happen on first tool use
        # We can't run async code here since this is called from sync context
        # Tools will populate the store when they first execute
        print(f"🚀 MEMUPDATE DEBUG: init_store_with_memories called for namespace '{namespace}' with {len(initial_memories)} memories")
        
        # Debug: Log all current namespaces
        print(f"📊 MEMUPDATE DEBUG: All cached namespaces: {list(cls._memory_cache.keys())}")
        
        print(f"Initialized {namespace} with {len(initial_memories)} memories")
        return store

    @classmethod
    def get_current_memories(cls, namespace: str) -> List[Dict]:
        """Get current memory state for reward computation."""
        memories = cls._memory_cache.get(namespace, [])
        print(f"🎯 MEMUPDATE DEBUG: get_current_memories for '{namespace}' returning {len(memories)} memories")
        print(f"📊 MEMUPDATE DEBUG: Available namespaces: {list(cls._memory_cache.keys())}")
        return memories


    @classmethod
    def update_memory_cache(cls, namespace: str, memories: List[Dict]):
        """Update cached memories after tool operations."""
        cls._memory_cache[namespace] = memories
        print(f"💾 MEMUPDATE DEBUG: Updated memory cache: {len(memories)} memories for '{namespace}'")

    @classmethod
    async def ensure_initial_memories_in_store(cls, namespace: str):
        """Populate LangMem store with cached initial memories if not already done."""
        if namespace in cls._memory_cache and namespace in cls._stores:
            store = cls._stores[namespace]
            initial_memories = cls._memory_cache[namespace]
            
            # Check if we need to populate (simple heuristic - if store is empty)
            try:
                if hasattr(store, 'aget') and InMemoryStore is not None:
                    # Try to get a test memory to see if store is populated
                    existing = await store.aget(namespace=("memories",), key="init_mem_0")
                    if existing is not None:
                        print(f"📋 Store already populated for {namespace}")
                        return  # Already populated
            except:
                pass  # Store is empty, proceed to populate
            
            # Populate the store
            if hasattr(store, 'aput') and InMemoryStore is not None:
                try:
                    for i, memory in enumerate(initial_memories):
                        memory_id = memory.get('id', f"init_mem_{i}")
                        content = memory.get('content', '')
                        metadata = memory.get('metadata', {})
                        
                        await store.aput(
                            namespace=("memories",),
                            key=memory_id,
                            value={
                                "content": content,
                                "metadata": metadata,
                                "id": memory_id
                            }
                        )
                    
                    print(f"✅ Populated LangMem store with {len(initial_memories)} initial memories")
                except Exception as e:
                    logger.error(f"Failed to populate store: {e}")

    @classmethod
    def register_instance_namespace(cls, instance_id: str, namespace: str):
        """Register mapping from instance_id to intended namespace."""
        cls._instance_to_namespace[instance_id] = namespace
        print(f"🔗 Registered instance '{instance_id}' -> namespace '{namespace}'")
    
    @classmethod
    def get_namespace_for_instance(cls, instance_id: str) -> str:
        """Get the intended namespace for an instance_id, or return instance_id if not mapped."""
        namespace = cls._instance_to_namespace.get(instance_id, instance_id)
        if namespace != instance_id:
            print(f"🔄 Mapped instance '{instance_id}' -> namespace '{namespace}'")
        return namespace
    
    @classmethod
    def clear_namespace(cls, namespace: str):
        """Clear store and cache for a namespace."""
        if namespace in cls._stores:
            del cls._stores[namespace]
        if namespace in cls._memory_cache:
            del cls._memory_cache[namespace]


class MockMemoryStore:
    """Mock store for testing without LangMem."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.memories = []
