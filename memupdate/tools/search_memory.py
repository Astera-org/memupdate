"""Memory search tool - wraps LangMem search functionality in verl BaseTool interface."""

import logging
import os
from typing import Any, Optional

print("🔍 SearchMemoryTool: Attempting langmem imports...")
try:
    from langmem import create_search_memory_tool
    print("✅ SearchMemoryTool: create_search_memory_tool imported successfully")
    from langgraph.store.memory import InMemoryStore
    print("✅ SearchMemoryTool: InMemoryStore imported successfully")
except ImportError as e:
    print(f"❌ SearchMemoryTool: ImportError during langmem imports: {e}")
    # Fallback if langmem not available
    create_search_memory_tool = None
    InMemoryStore = None
except Exception as e:
    print(f"❌ SearchMemoryTool: Other error during langmem imports: {type(e).__name__}: {e}")
    create_search_memory_tool = None
    InMemoryStore = None

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionToolSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    ToolResponse,
)

logger = logging.getLogger(__name__)

# Set up debug logging to file if requested
def _debug_log(message: str):
    """Log debug message to file if MEMUPDATE_TOOL_DEBUG is set."""
    if os.getenv('MEMUPDATE_TOOL_DEBUG'):
        log_file = os.getenv('MEMUPDATE_LOG_FILE', '/workspace/memupdate/tool_debug.log')
        try:
            with open(log_file, 'a') as f:
                f.write(f"[SearchMemoryTool] {message}\n")
                f.flush()
        except:
            pass
    print(f"[TOOL] {message}")


class SearchMemoryTool(BaseTool):
    """Memory search tool that wraps LangMem search functionality."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        _debug_log("🔧 Initializing SearchMemoryTool...")
        
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        
        # Use shared memory store manager
        from .base_memory_tool import MemoryStoreManager
        self.store_manager = MemoryStoreManager
        
        # Initialize LangMem components
        self.langmem_search = None
        self._init_langmem()
        
        _debug_log(f"✅ SearchMemoryTool initialized (LangMem available: {self.langmem_search is not None})")

    def _init_langmem(self):
        """Initialize LangMem components."""
        if create_search_memory_tool is None or InMemoryStore is None:
            _debug_log("❌ LangMem imports not available (langmem or langgraph missing)")
            logger.warning("LangMem not available, using mock implementation")
            return
            
        try:
            _debug_log("💾 Using shared MemoryStoreManager instead of creating separate store")
            
            # 🔧 CRITICAL FIX: Don't create separate store - use shared MemoryStoreManager
            # Each tool creating its own InMemoryStore() causes isolation!
            # We'll create LangMem tools with shared stores during execute()
            
            # 🔧 FIX: Set langmem_search to indicate LangMem is available 
            self.langmem_search = "available"  # Placeholder to indicate LangMem is working
            
            _debug_log("✅ Will use shared store from MemoryStoreManager per namespace")
            print("✅ SearchMemoryTool will use shared stores per namespace")
        except Exception as e:
            _debug_log(f"❌ LangMem search tool initialization failed: {e}")
            logger.error(f"Failed to initialize LangMem search tool: {e}")
            self.langmem_search = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory search."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search_memory",
                description="Search and retrieve relevant memories from the memory database",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Natural language query to search for relevant memories"
                        ),
                        "top_k": OpenAIFunctionPropertySchema(
                            type="integer", 
                            description="Number of most relevant memories to retrieve (default: 5)"
                        ),
                        "memory_type": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Type of memory to search: episodic, semantic, or procedural",
                            enum=["episodic", "semantic", "procedural"]
                        ),
                        "threshold": OpenAIFunctionPropertySchema(
                            type="number",
                            description="Minimum similarity threshold for retrieved memories (0-1)"
                        )
                    },
                    required=["query"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a search tool instance with initial memory loading."""
        from uuid import uuid4
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize memory store with initial memories if provided
        initial_memories = kwargs.get('initial_memories', [])
        namespace = kwargs.get('namespace', instance_id)
        
        print(f"🔍 MEMUPDATE DEBUG: SearchMemoryTool.create called with namespace='{namespace}', instance_id='{instance_id}'")
        
        # Check if there's a create_kwargs that contains the actual namespace and initial_memories
        create_kwargs = kwargs.get('create_kwargs', {})
        if 'namespace' in create_kwargs:
            actual_namespace = create_kwargs['namespace']
            print(f"🔍 MEMUPDATE DEBUG: Found actual namespace in create_kwargs: '{actual_namespace}'")
            namespace = actual_namespace
            
        # 🔧 CRITICAL FIX: Also check for initial_memories in create_kwargs
        if 'initial_memories' in create_kwargs:
            actual_initial_memories = create_kwargs['initial_memories']
            print(f"🔍 MEMUPDATE DEBUG: Found {len(actual_initial_memories)} initial_memories in create_kwargs")
            initial_memories = actual_initial_memories
        
        print(f"🔍 MEMUPDATE DEBUG: Final values - namespace='{namespace}', initial_memories count={len(initial_memories)}")
        
        # 🔧 CRITICAL FIX: Register the mapping from instance_id to intended namespace
        if namespace != instance_id:
            self.store_manager.register_instance_namespace(instance_id, namespace)
        
        if initial_memories:
            self.store_manager.init_store_with_memories(namespace, initial_memories)
            return instance_id, ToolResponse(text=f"Memory search tool initialized with {len(initial_memories)} memories in namespace '{namespace}'")
        else:
            self.store_manager.get_or_create_store(namespace)  # Ensure store exists
            return instance_id, ToolResponse(text=f"Memory search tool initialized with empty store in namespace '{namespace}'")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory search operation."""
        _debug_log(f"🔍 SearchMemoryTool.execute called with query: {parameters.get('query', 'N/A')}")
        try:
            print(f"🔍 MEMUPDATE DEBUG: SearchMemoryTool.execute kwargs: {kwargs}")
            
            # Get namespace from kwargs
            namespace = kwargs.get("namespace", instance_id)
            
            # Check if there's an execute_kwargs that contains the actual namespace
            execute_kwargs = kwargs.get('execute_kwargs', {})
            if 'namespace' in execute_kwargs:
                actual_namespace = execute_kwargs['namespace']
                print(f"🔍 MEMUPDATE DEBUG: Found actual namespace in execute_kwargs: '{actual_namespace}'")
                namespace = actual_namespace
            
            # 🔧 CRITICAL FIX: Use mapped namespace if available
            namespace = self.store_manager.get_namespace_for_instance(namespace)
            
            query = parameters.get("query", "")
            top_k = parameters.get("top_k", 5)
            memory_type = parameters.get("memory_type", None)
            threshold = parameters.get("threshold", 0.0)
            
            _debug_log(f"🔍 SearchMemoryTool.execute called with namespace='{namespace}', query='{query}'")

            if not query:
                return ToolResponse(text="Error: Query is required for memory search"), 0.0, {}

            # 🔧 CRITICAL FIX: Use Ray Actor method directly to avoid serialization issues
            # Getting local store copies via ray.get() creates separate instances!
            
            if create_search_memory_tool is not None and InMemoryStore is not None:
                try:
                    print(f"🔍 DEBUG: About to search with query: '{query[:50]}...'")
                    print(f"🔍 DEBUG: Search parameters: top_k={top_k}, memory_type={memory_type}")
                    
                    # Use Ray Actor method directly
                    result = self.store_manager.search_memory_via_actor(namespace, query, top_k)
                    
                    if result["success"]:
                        search_results = result["results"]
                        total_count = result["total_count"]
                        
                        print(f"🔍 DEBUG: Ray Actor search result: found {len(search_results)} memories")
                        print(f"🔍 DEBUG: Total memories in namespace '{namespace}': {total_count}")
                        
                        if search_results:
                            # Format results similar to LangMem output
                            formatted_results = "Found some relevant memories:\n"
                            for i, mem in enumerate(search_results, 1):
                                content = mem.get("content", "")[:200] + ("..." if len(mem.get("content", "")) > 200 else "")
                                formatted_results += f"{i}. {content}\n"
                                
                            return ToolResponse(
                                text=formatted_results
                            ), 0.1, {"memories_found": len(search_results), "total_memories": total_count}
                        else:
                            return ToolResponse(
                                text=f"No memories found matching '{query}'. Total memories available: {total_count}"
                            ), 0.1, {"memories_found": 0, "total_memories": total_count}
                    else:
                        return ToolResponse(text=f"Failed to search memories: {result.get('error', 'Unknown error')}", 0.0, {})
                    
                except Exception as e:
                    logger.error(f"LangMem search failed: {e}")
                    return ToolResponse(text=f"Memory search failed: {str(e)}"), 0.0, {}
            else:
                # Mock implementation when LangMem not available
                mock_memories = [
                    f"Mock memory 1 related to: {query}",
                    f"Mock memory 2 related to: {query}",
                ][:top_k]
                
                result_text = f"Mock search results for '{query}':\n" + "\n".join(
                    f"{i+1}. {memory}" for i, memory in enumerate(mock_memories)
                )
                
                return ToolResponse(text=result_text), 0.1, {"memories_found": len(mock_memories)}

        except Exception as e:
            logger.error(f"Memory search execution failed: {e}")
            return ToolResponse(text=f"Memory search failed: {str(e)}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        """Release tool instance but preserve memory state for reward computation."""
        # 🔧 CRITICAL: Don't clear namespace here - memory state needed for reward computation
        # MemoryStoreManager persists across tool instances using class-level storage
        namespace = kwargs.get("namespace", instance_id)
        
        # 🔧 CRITICAL FIX: Use mapped namespace if available
        namespace = self.store_manager.get_namespace_for_instance(namespace)
        
        _debug_log(f"🧹 SearchMemoryTool.release called for namespace '{namespace}' (memory preserved)")
        
        # Just log current memory state for debugging
        current_memories = self.store_manager.get_current_memories(namespace)
        _debug_log(f"📊 Memory state at release: {len(current_memories)} memories in '{namespace}'")
        
        # Return success (no actual cleanup needed since MemoryStoreManager handles persistence)
        return f"Released SearchMemoryTool instance for namespace '{namespace}'"