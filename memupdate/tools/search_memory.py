"""Memory search tool - wraps LangMem search functionality in verl BaseTool interface."""

import logging
import os
from typing import Any, Optional

try:
    from langmem import create_search_memory_tool
    from langgraph.store.memory import InMemoryStore
except ImportError:
    # Fallback if langmem not available
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
    logger.info(f"[TOOL] {message}")


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
            _debug_log("💾 Creating InMemoryStore (text-based, no embeddings)...")
            
            # Create simple text-based store without embeddings
            # This uses exact string matching which is sufficient for our use case
            self.store = InMemoryStore()
            
            _debug_log("✅ InMemoryStore created successfully (text-based)")
            logger.info("Using text-based InMemoryStore (no embeddings required)")
            
            # Create LangMem search tool
            _debug_log("🛠️  Creating LangMem search tool...")
            self.langmem_search = create_search_memory_tool(
                namespace=("memories",),
                store=self.store
            )
            _debug_log("✅ LangMem search tool created successfully")
            logger.info("LangMem search tool initialized successfully")
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
            # Get namespace from kwargs
            namespace = kwargs.get("namespace", instance_id)
            
            query = parameters.get("query", "")
            top_k = parameters.get("top_k", 5)
            memory_type = parameters.get("memory_type", None)
            threshold = parameters.get("threshold", 0.0)

            if not query:
                return ToolResponse(text="Error: Query is required for memory search"), 0.0, {}

            # Get shared store for this namespace
            store = self.store_manager.get_or_create_store(namespace)

            # If LangMem is available, use it
            if self.langmem_search is not None:
                try:
                    # Use LangMem search tool
                    result = await self.langmem_search.ainvoke({
                        "query": query,
                        "k": top_k
                    })
                    
                    # Format the results
                    if result and hasattr(result, 'content'):
                        search_results = result.content
                    else:
                        search_results = str(result) if result else "No memories found"
                        
                    return ToolResponse(
                        text=f"Found {len(result) if isinstance(result, list) else 'some'} relevant memories:\n{search_results}"
                    ), 0.1, {"memories_found": len(result) if isinstance(result, list) else 0}
                    
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