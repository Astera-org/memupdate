"""Shared memory store management for all tools."""

import logging
from typing import Dict, List, Optional

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
                logger.info(f"Created LangMem store for namespace: {namespace}")

        return cls._stores[namespace]

    @classmethod
    def init_store_with_memories(cls, namespace: str, initial_memories: List[Dict]):
        """Initialize a store with pre-existing memories."""
        store = cls.get_or_create_store(namespace)

        # Cache the memories for reward computation
        cls._memory_cache[namespace] = initial_memories.copy()

        # Add to LangMem store (would use manage_memory tool in reality)
        for memory in initial_memories:
            # Store initialization logic here
            pass

        logger.info(f"Initialized {namespace} with {len(initial_memories)} memories")
        return store

    @classmethod
    def get_current_memories(cls, namespace: str) -> List[Dict]:
        """Get current memory state for reward computation."""
        return cls._memory_cache.get(namespace, [])

    @classmethod
    def update_memory_cache(cls, namespace: str, memories: List[Dict]):
        """Update cached memories after tool operations."""
        cls._memory_cache[namespace] = memories

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
