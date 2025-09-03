"""Shared memory store management for all tools."""

import ray
import os
import uuid
import logging
from typing import Dict, List, Optional

try:
    from langgraph.store.memory import InMemoryStore
except ImportError:
    InMemoryStore = None

logger = logging.getLogger(__name__)


@ray.remote
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
        # namespace/conversation_id -> InMemoryStore instance
        self._stores: Dict[str, InMemoryStore] = {}
        # namespace -> initial memories for reward computation
        self._initial_memories: Dict[str, List[Dict]] = {}
        # namespace -> current memory state cache
        self._memory_cache: Dict[str, List[Dict]] = {}
        
        print(f"🏢 MemoryBrokerActor initialized in process {os.getpid()}")
    
    async def get_store_for_tools(self, namespace: str) -> InMemoryStore:
        """Get InMemoryStore for tool execution on rollout workers."""
        if namespace not in self._stores:
            self._stores[namespace] = InMemoryStore()
            print(f"✅ Created new InMemoryStore for namespace: {namespace}")
        else:
            print(f"🔄 Using existing InMemoryStore for namespace: {namespace}")
        return self._stores[namespace]
    
    async def create_memory_in_store(self, namespace: str, memory_data: dict) -> dict:
        """Create memory directly in Ray Actor store (avoids serialization issues)."""
        if namespace not in self._stores:
            self._stores[namespace] = InMemoryStore()
            print(f"✅ Created new InMemoryStore for namespace: {namespace}")
        
        store = self._stores[namespace]
        
        # Use LangMem's create_manage_memory_tool to add the memory
        try:
            from langmem import create_manage_memory_tool
            langmem_manage = create_manage_memory_tool(
                namespace=("memories",),
                store=store
            )
            
            result = await langmem_manage.ainvoke(memory_data)
            
            # Return updated count and result
            memories = await store.asearch(("memories",), query="", limit=1000)
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
        if namespace not in self._stores:
            return {"result": "Namespace not found", "success": False}
        
        store = self._stores[namespace]
        
        try:
            from langmem import create_manage_memory_tool
            langmem_manage = create_manage_memory_tool(
                namespace=("memories",),
                store=store
            )
            
            result = await langmem_manage.ainvoke(memory_data)
            
            memories = await store.asearch(("memories",), query="", limit=1000)
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
        if namespace not in self._stores:
            return {"results": [], "success": False}
        
        store = self._stores[namespace]
        
        try:
            memories = await store.asearch(("memories",), query=query, limit=limit)
            results = []
            for mem in memories:
                results.append({
                    "id": mem.key,
                    "content": mem.value.get("content", ""),
                    "metadata": mem.value.get("metadata", {}),
                })
            
            return {
                "results": results,
                "total_count": len(await store.asearch(("memories",), query="", limit=1000)),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error searching memory in Ray Actor store: {e}")
            return {
                "results": [],
                "success": False
            }
    
    async def init_conversation_memory(self, namespace: str, initial_memories: List[Dict]) -> str:
        """Initialize memory for a conversation (called per batch item)."""
        # 🔧 CRITICAL FIX: Don't reinitialize if store already exists (prevents overwriting tool changes)
        if namespace in self._stores:
            existing_count = len(await self.get_final_memories_for_reward(namespace))
            initial_count = len(initial_memories)
            print(f"🔄 Store already exists for namespace {namespace} with {existing_count} memories (vs {initial_count} initial), skipping reinitialization to preserve tool changes")
            return namespace
            
        print(f"🆕 Creating NEW store for namespace {namespace} with {len(initial_memories)} initial memories")
        store = InMemoryStore()
        self._stores[namespace] = store
        self._initial_memories[namespace] = initial_memories.copy()
        
        # Populate the LangMem store with initial memories
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
            
            await store.aput(
                namespace=("memories",),
                key=langmem_id,
                value={
                    "content": content,
                    "metadata": metadata,
                    "id": langmem_id
                }
            )
        
        self._memory_cache[namespace] = initial_memories.copy()
        print(f"✅ Successfully initialized NEW conversation memory for {namespace} with {len(initial_memories)} memories")
        return namespace
    
    async def get_final_memories_for_reward(self, namespace: str) -> List[Dict]:
        """Get final memory state for reward computation (called on reward workers)."""
        if namespace not in self._stores:
            print(f"⚠️  No memories found for namespace {namespace}")
            return []
        
        store = self._stores[namespace]
        
        try:
            print(f"🔍 DEBUG: About to query store for namespace {namespace} - store type: {type(store)}")
            # Query all memories from the actual LangMem store
            memories = await store.asearch(
                ("memories",),  # namespace_prefix as positional argument
                query="",  # Empty query returns all memories
                limit=1000  # High limit to get all memories
            )
            
            print(f"🔍 DEBUG: Store query successful - returned {len(memories)} memories")
            
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
            
            print(f"🎯 Retrieved {len(result)} final memories for reward computation from {namespace}")
            return result
            
        except Exception as e:
            print(f"❌ CRITICAL: Error querying memories for {namespace}: {type(e).__name__}: {e}")
            print(f"❌ FALLBACK: Using initial memories ({len(self._initial_memories.get(namespace, []))}) instead of actual store contents")
            # Fallback to initial memories
            return self._initial_memories.get(namespace, [])
    
    async def cleanup_conversation(self, namespace: str):
        """Clean up memory for a conversation after episode completion."""
        if namespace in self._stores:
            del self._stores[namespace]
        if namespace in self._initial_memories:
            del self._initial_memories[namespace]
        if namespace in self._memory_cache:
            del self._memory_cache[namespace]
        print(f"🧹 Cleaned up memory for namespace: {namespace}")


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
                print(f"📡 Connected to existing MemoryBrokerActor from process {os.getpid()}")
                
            except ValueError:
                # Actor doesn't exist, create it
                try:
                    cls._broker_actor = MemoryBrokerActor.options(
                        name="memory_broker",
                        lifetime="detached"  # Survives driver failures
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
    def init_store_with_memories(cls, namespace: str, initial_memories: List[Dict]):
        """Initialize store with memories (called once per batch item)."""
        broker = cls.get_broker_actor()
        ray.get(broker.init_conversation_memory.remote(namespace, initial_memories))

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
        print(f"🔗 Mapped instance '{instance_id}' -> namespace '{namespace}' in process {os.getpid()}")
    
    @classmethod
    def get_namespace_for_instance(cls, instance_id: str) -> str:
        """Get namespace for instance_id (request_id)."""
        namespace = cls._instance_to_namespace.get(instance_id, instance_id)
        if namespace != instance_id:
            print(f"🔄 Using mapped namespace '{namespace}' for instance '{instance_id}'")
        return namespace
    
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
