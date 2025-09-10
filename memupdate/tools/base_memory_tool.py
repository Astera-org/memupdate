"""Shared memory store management for all tools."""

import ray
import os
import uuid
import logging
# torch not needed for CPU-only embeddings
from typing import Dict, List, Optional, Any

try:
    from langgraph.store.memory import InMemoryStore
    # IndexConfig not needed for CPU-only cached embeddings
except ImportError:
    InMemoryStore = None

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class MemoryBrokerActor:
    """
    Ray Actor that serves as memory broker between rollout workers and reward workers.
    
    Key Design:
    - Shared across ALL Ray workers in the verl cluster
    - Manages InMemoryStore instances per conversation namespace
    - Handles concurrent access from rollout workers (write) and reward workers (read)
    - Provides memory isolation per batch item
    """
    
    def __init__(self):
        # namespace -> InMemoryStore instance
        self._stores: Dict[str, InMemoryStore] = {}
        
        # Centralized storage for conversation memories by sample_id
        # sample_id (e.g., "conv-48") -> initial memories from LoCoMo dataset
        self._conversation_memories: Dict[str, List[Dict]] = {}
        
        # Cached embeddings (key: content hash, value: embedding with metadata)
        self._embedding_cache = {}
        self._load_embedding_cache()
        self._load_conversation_memories()
        
        # Single shared embedding model for generating new embeddings
        self._embedding_model = None
        self._init_embedding_model()
        
        print(f"🏢 MemoryBrokerActor initialized with {len(self._conversation_memories)} conversations in process {os.getpid()}")
    
    def _load_embedding_cache(self):
        """Load cached embeddings with metadata."""
        try:
            import pickle
            import numpy as np
            
            cache_file = "/workspace/memupdate/data/embedding_cache/memory_embeddings.pkl"
            
            with open(cache_file, 'rb') as f:
                self._embedding_cache = pickle.load(f)
            
            # Validate embeddings
            for key, value in self._embedding_cache.items():
                embedding = value.get('embedding')
                if isinstance(embedding, np.ndarray):
                    norm = np.linalg.norm(embedding)
                    if norm == 0 or np.isnan(norm):
                        print(f"⚠️ Bad embedding for key {key}: norm={norm}")
                        value['embedding'] = np.random.randn(1024) * 0.01
            
            # Count embeddings per conversation
            conv_counts = {}
            for key, value in self._embedding_cache.items():
                sid = value.get('sample_id')
                if sid:
                    conv_counts[sid] = conv_counts.get(sid, 0) + 1
            
            print(f"💾 Loaded {len(self._embedding_cache)} embeddings for {len(conv_counts)} conversations")
            if conv_counts:
                print(f"📊 Sample distribution: {list(conv_counts.items())[:3]}...")
                
        except Exception as e:
            print(f"⚠️ Failed to load embedding cache: {e}")
            self._embedding_cache = {}
    
    def _load_conversation_memories(self):
        """Load all conversation memories from LoCoMo dataset at startup."""
        try:
            import json
            from pathlib import Path
            
            locomo_path = Path("/workspace/locomo/data/locomo10.json")
            if not locomo_path.exists():
                print(f"⚠️ LoCoMo data not found at {locomo_path}, skipping conversation memory loading")
                return
            
            with open(locomo_path, 'r') as f:
                locomo_data = json.load(f)
            
            # Process each conversation
            for conv in locomo_data:
                sample_id = conv.get("sample_id", "unknown")
                observation = conv.get("observation", {})
                
                # Convert observations to memory format (same logic as preprocess_locomo.py)
                memories = self._convert_observation_to_memories(observation)
                
                self._conversation_memories[sample_id] = memories
                print(f"  📚 Loaded {len(memories)} memories for {sample_id}")
            
            print(f"✅ Loaded conversation memories for {len(self._conversation_memories)} conversations")
            
        except Exception as e:
            print(f"❌ Failed to load conversation memories: {e}")
            import traceback
            traceback.print_exc()
    
    def _convert_observation_to_memories(self, observations: Dict) -> List[Dict]:
        """Convert conversation observations to initial memory entries."""
        memories = []
        
        # LoCoMo observations are nested under session keys
        for session_key, session_obs in observations.items():
            if isinstance(session_obs, dict):
                for speaker, speaker_observation in session_obs.items():
                    for fact_entry in speaker_observation:
                        if isinstance(fact_entry, list) and len(fact_entry) >= 2:
                            fact_text, evidence = fact_entry[0], fact_entry[1]
                            
                            memory = {
                                "content": fact_text,
                                "speaker": speaker,
                                "evidence": evidence,
                                "session": session_key,
                                "memory_type": "episodic",
                                "source": "conversation_observations",
                            }
                            memories.append(memory)
                            # TODO: examine memory format
                            # breakpoint()
        
        return memories
    
    def _init_embedding_model(self):
        """Initialize shared embedding model for generating new embeddings."""
        try:
            from memupdate.data.qwen_embeddings import QwenEmbeddings
            
            # Load once and share across all samples
            self._embedding_model = QwenEmbeddings()
            print(f"🚀 Loaded shared embedding model in MemoryBrokerActor")
                
        except Exception as e:
            print(f"⚠️ Failed to initialize shared embedding model: {e}")
            self._embedding_model = None
    
    def get_initial_memories(self, sample_id: str) -> List[Dict]:
        """Get initial memories for a sample_id from loaded conversation data."""
        memories = self._conversation_memories.get(sample_id, [])
        if not memories:
            print(f"⚠️ No conversation memories found for sample_id '{sample_id}'")
        return memories.copy()  # Return a copy to prevent modification
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded conversations for verification."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024
        except:
            memory_mb = 0
            
        return {
            "total_conversations": len(self._conversation_memories),
            "sample_ids": list(self._conversation_memories.keys()),
            "memory_counts": {sid: len(mems) for sid, mems in self._conversation_memories.items()},
            "active_stores": len(self._stores),
            "memory_usage_mb": memory_mb,
            "store_namespaces": list(self._stores.keys())[:10]  # Show first 10 to avoid huge output
        }
    
    def _get_sample_id_from_namespace(self, namespace: str) -> str:
        """Extract sample_id from namespace (e.g., 'conv-48-qa2-abc123' -> 'conv-48')."""
        if '-qa' in namespace:
            return namespace.split('-qa')[0]
        return namespace
    
    def _create_store_with_embeddings(self, namespace: str) -> InMemoryStore:
        """Create InMemoryStore with smart cached embeddings."""
        try:
            from .cached_embeddings import SmartCachedEmbeddings
            import torch
            
            # Extract sample_id from namespace for filtering
            sample_id = self._get_sample_id_from_namespace(namespace)
            
            # Create smart embeddings that:
            # 1. Uses our already-loaded cache
            # 2. Filters by conversation  
            # 3. Uses shared embedding model (no per-sample initialization!)
            embeddings = SmartCachedEmbeddings(
                cache=self._embedding_cache,  # Use already-loaded cache
                sample_id=sample_id,  # Filter by conversation
                embedding_model=self._embedding_model,  # Use shared model
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Get stats for logging
            stats = embeddings.get_stats()
            print(f"🧠 Created embeddings for {namespace}: {stats['cached_embeddings']} cached for {sample_id}, shared_model={stats['has_model']}")
            
            index_config = {
                "embed": embeddings,
                "dims": 1024,  # Qwen3-0.6B dimension
                "fields": ["content"],
            }
            
            store = InMemoryStore(index=index_config)
            
            return store
            
        except Exception as e:
            print(f"❌ Failed to create store with embeddings: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic store
            store = InMemoryStore()
            print(f"⚠️ Created basic InMemoryStore (no embeddings) for namespace '{namespace}'")
            return store
    
    async def get_store_for_tools(self, namespace: str) -> InMemoryStore:
        """Get InMemoryStore for tool execution on rollout workers."""
        if namespace not in self._stores:
            # 🔧 CRITICAL FIX: Create stores with embeddings from the start
            store = self._create_store_with_embeddings(namespace)
            self._stores[namespace] = store
            print(f"📝 Created new store for namespace: {namespace} (active stores: {len(self._stores)})")
        else:
            print(f"🔄 Using existing InMemoryStore for namespace: {namespace} (active stores: {len(self._stores)})")
        
        return self._stores[namespace]
    
    async def create_memory_in_store(self, namespace: str, memory_data: dict) -> dict:
        """Create memory directly in Ray Actor store (avoids serialization issues)."""
        # TODO: validate langmem commands are correct: https://langchain-ai.github.io/langmem/reference/memory/#langmem.create_memory_store_manager
        
        if namespace not in self._stores:
            store = self._create_store_with_embeddings(namespace)
            self._stores[namespace] = store
        
        store = self._stores[namespace]
        
        # Use LangMem's create_manage_memory_tool to add the memory
        try:
            from langmem import create_manage_memory_tool
            langmem_manage = create_manage_memory_tool(
                namespace=("memories",),
                store=store
            )
            
            # 🔧 CRITICAL FIX: LangMem expects {content, action, id} format, not our dict format
            langmem_params = {
                "content": memory_data.get("content", ""),
                "action": memory_data.get("action", "create")
            }
            if "id" in memory_data:
                langmem_params["id"] = memory_data["id"]
            
            result = await langmem_manage.ainvoke(langmem_params)
            
            # Return updated count and result
            memories = await store.asearch(("memories",), query="", limit=999999)
            return {
                "result": result,
                "new_count": len(memories),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error creating memory in Ray Actor store: {e}")
            return {
                "result": str(e),
                "new_count": 0,
                "success": False
            }
    
    async def update_memory_in_store(self, namespace: str, memory_data: dict) -> dict:
        """Update memory directly in Ray Actor store."""
        # TODO: validate langmem commands are correct: https://langchain-ai.github.io/langmem/reference/memory/#langmem.create_memory_store_manager
        if namespace not in self._stores:
            # Create store with embeddings if it doesn't exist (rare case)
            print(f"🔧 Creating new store with embeddings for namespace '{namespace}' during update")
            store = self._create_store_with_embeddings(namespace)
            self._stores[namespace] = store
        
        store = self._stores[namespace]
        
        try:
            from langmem import create_manage_memory_tool
            langmem_manage = create_manage_memory_tool(
                namespace=("memories",),
                store=store
            )
            
            # 🔧 CRITICAL FIX: LangMem expects {content, action, id} format, not our dict format
            langmem_params = {
                "content": memory_data.get("content", ""),
                "action": memory_data.get("action", "update")
            }
            if "id" in memory_data:
                langmem_params["id"] = memory_data["id"]
            
            result = await langmem_manage.ainvoke(langmem_params)
            
            memories = await store.asearch(("memories",), query="", limit=999999)
            return {
                "result": result,
                "new_count": len(memories),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error updating memory in Ray Actor store: {e}")
            return {
                "result": str(e),
                "success": False
            }
    
    async def search_memory_in_store(self, namespace: str, query: str, limit: int = 10) -> dict:
        """Search memory directly in Ray Actor store."""
        # TODO: validate langmem commands are correct: https://langchain-ai.github.io/langmem/reference/memory/#langmem.create_memory_store_manager
        if namespace not in self._stores:
            print(f"⚠️ Namespace '{namespace}' not found in stores - creating empty store for search")
            # Create an empty store so search can work (even if it returns no results)
            store = self._create_store_with_embeddings(namespace)
            self._stores[namespace] = store
            # Return empty results since store is new
            return {"results": [], "success": True}
        
        store = self._stores[namespace]
        
        try:
            has_embeddings = hasattr(store, 'index_config') and store.index_config is not None
            memories = await store.asearch(("memories",), query=query, limit=limit)

            results = []
            for mem in memories:
                results.append({
                    "id": mem.key,
                    "content": mem.value.get("content", ""),
                    "metadata": mem.value.get("metadata", {}),
                })
            # TODO: breakpoint to see outcome of search
            # breakpoint()
            
            return {
                "results": results,
                "total_count": len(await store.asearch(("memories",), query="", limit=999999)),
                "success": True,
                "semantic_search": has_embeddings
            }
            
        except Exception as e:
            print(f"❌ Error searching memory in Ray Actor store: {e}")
            import traceback
            traceback.print_exc()
            return {
                "results": [],
                "success": False
            }
    
    
    async def init_conversation_memory(self, namespace: str, sample_id: str) -> str:
        """Initialize memory for a conversation using sample_id (called per batch item).
        
        Args:
            namespace: Unique namespace for this trajectory (e.g., "conv-48-qa2-abc123")
            sample_id: Conversation identifier to load initial memories from (e.g., "conv-48")
        """
        # Get initial memories from loaded conversation data
        initial_memories = self.get_initial_memories(sample_id)    
        
        # Create store with embeddings
        store = self._create_store_with_embeddings(namespace)
        self._stores[namespace] = store
        
        # Store the sample_id -> namespace mapping for later retrieval
        # This allows reward manager to get initial memories using sample_id
        # Note: We don't store initial_memories per namespace anymore, just reference the sample_id
        
        # Populate store with initial memories
        # TODO: check what this is doing
        for i, memory in enumerate(initial_memories):
            memory_id = memory.get('id', f"init_mem_{i}")
            content = memory.get('content', '')
            metadata = memory.get('metadata', {})
            
            # Ensure UUID format for memory IDs
            try:
                uuid.UUID(memory_id)
                langmem_id = memory_id
            except ValueError:
                langmem_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))
            
            try:
                await store.aput(
                    namespace=("memories",),
                    key=langmem_id,
                    value={
                        "content": content,
                        "metadata": metadata,
                        "id": langmem_id
                    }
                )
            except Exception as e:
                print(f"❌ Failed to add initial memory {memory_id}: {e}")
            # TODO: debug memory store, what's in there in what format
            # breakpoint()
        
        return namespace

    async def get_final_memories_for_reward(self, namespace: str) -> List[Dict]:
        """Get final memory state for reward computation (called on reward workers)."""
        if namespace not in self._stores:
            print(f"⚠️  Namespace '{namespace}' not found - creating store (likely called during tool initialization)")
            # Return empty list if namespace was never created yet
            return []
        
        store = self._stores[namespace]
        
        try:
            # Query all memories from the actual LangMem store
            memories = await store.asearch(
                ("memories",),  # namespace_prefix as positional argument
                query="",  # Empty query returns all memories
                limit=999999  # High limit to get all memories
            )
        
            
            # Convert to reward manager format
            result = []
            for mem in memories:
                result.append({
                    "id": mem.key,
                    "content": mem.value.get("content", ""),
                    "metadata": mem.value.get("metadata", {}),
                    "namespace": namespace,
                    "created_at": mem.created_at.isoformat() if mem.created_at else None,
                    "updated_at": mem.updated_at.isoformat() if mem.updated_at else None,
                })
            
            return result
            
        except Exception as e:
            print(f"❌ CRITICAL: Error querying memories for {namespace}: {type(e).__name__}: {e}")
    
    async def cleanup_conversation(self, namespace: str):
        """Clean up memory for a conversation after episode completion."""
        if namespace in self._stores:
            # Simple cleanup - the original approach
            del self._stores[namespace]
            
            # Force garbage collection to help with memory cleanup
            import gc
            gc.collect()


class MemoryStoreManager:
    """
    Updated MemoryStoreManager that uses Ray Actor for cross-process memory sharing.
    
    This class is instantiated on both:
    - Rollout workers (for tool execution)
    - Reward workers (for reward computation)
    """
    
    _broker_actor: Optional[ray.actor.ActorHandle] = None
    _instance_to_namespace: Dict[str, str] = {}

    @classmethod
    def get_broker_actor(cls) -> ray.actor.ActorHandle:
        """Get or create the shared memory broker actor."""
        if cls._broker_actor is None:
            try:
                # First check if Ray is initialized
                if not ray.is_initialized():
                    raise RuntimeError("Ray is not initialized - cannot create memory broker actor")
                
                # Check if actor already exists in Ray cluster
                cls._broker_actor = ray.get_actor("memory_broker")
                
            except ValueError:
                # Actor doesn't exist, create it
                try:
                    cls._broker_actor = MemoryBrokerActor.options(
                        name="memory_broker",
                        lifetime="detached",  # Survives driver failures
                    ).remote()
                    print(f"🏢 Created new MemoryBrokerActor from process {os.getpid()}")
                except Exception as create_error:
                    # Handle race condition - another worker might have created it
                    if "already exists" in str(create_error):
                        try:
                            cls._broker_actor = ray.get_actor("memory_broker")
                            print(f"📡 Connected to MemoryBrokerActor created by another worker from process {os.getpid()}")
                        except Exception as get_error:
                            raise RuntimeError(f"Failed to create or connect to MemoryBrokerActor: {get_error}")
                    else:
                        raise RuntimeError(f"Failed to create MemoryBrokerActor: {create_error}")
                        
        return cls._broker_actor
    
    @classmethod
    def get_or_create_store(cls, namespace: str) -> InMemoryStore:
        """Get InMemoryStore for namespace (used by tools on rollout workers)."""
        broker = cls.get_broker_actor()
        return ray.get(broker.get_store_for_tools.remote(namespace))
    
    @classmethod
    def get_initial_memories(cls, sample_id: str) -> List[Dict]:
        """Get initial memories for a sample_id from the broker."""
        broker = cls.get_broker_actor()
        return ray.get(broker.get_initial_memories.remote(sample_id))
    
    @classmethod
    def get_conversation_stats(cls) -> Dict[str, Any]:
        """Get conversation statistics from the broker."""
        broker = cls.get_broker_actor()
        return ray.get(broker.get_conversation_stats.remote())

    @classmethod
    def get_current_memories(cls, namespace: str) -> List[Dict]:
        """Get current memory state for reward computation (called on reward workers)."""
        broker = cls.get_broker_actor()
        return ray.get(broker.get_final_memories_for_reward.remote(namespace))
    
    @classmethod
    def create_memory_via_actor(cls, namespace: str, memory_data: dict) -> dict:
        """Create memory directly in Ray Actor store (fixes serialization issue)."""
        broker = cls.get_broker_actor()
        return ray.get(broker.create_memory_in_store.remote(namespace, memory_data))
    
    @classmethod
    def update_memory_via_actor(cls, namespace: str, memory_data: dict) -> dict:
        """Update memory directly in Ray Actor store (fixes serialization issue)."""
        broker = cls.get_broker_actor()
        return ray.get(broker.update_memory_in_store.remote(namespace, memory_data))
    
    @classmethod
    def search_memory_via_actor(cls, namespace: str, query: str, limit: int = 10) -> dict:
        """Search memory directly in Ray Actor store (fixes serialization issue)."""
        broker = cls.get_broker_actor()
        return ray.get(broker.search_memory_in_store.remote(namespace, query, limit))

    @classmethod
    def register_instance_namespace(cls, instance_id: str, namespace: str):
        """Register mapping from instance_id (request_id) to conversation namespace."""
        cls._instance_to_namespace[instance_id] = namespace
    
    @classmethod
    def get_namespace_for_instance(cls, instance_id: str) -> str:
        """Get namespace for instance_id (request_id)."""
        namespace = cls._instance_to_namespace.get(instance_id, instance_id)
        return namespace
    
    @classmethod
    def init_conversation_memory(cls, namespace: str, sample_id: str) -> bool:
        """Ensure store is initialized, but only if it doesn't exist (idempotent)."""
        broker = cls.get_broker_actor()
        return ray.get(broker.init_conversation_memory.remote(namespace, sample_id))
    
    @classmethod
    def cleanup_conversation(cls, namespace: str):
        """Clean up memory after episode completion."""
        broker = cls.get_broker_actor()
        ray.get(broker.cleanup_conversation.remote(namespace))


class MockMemoryStore:
    """Mock store for testing without LangMem."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.memories = []
