"""Memory management tool - wraps LangMem management functionality in verl BaseTool interface."""

import logging
import os
from typing import Any, Optional

try:
    from langmem import create_manage_memory_tool
    from langgraph.store.memory import InMemoryStore
except ImportError:
    # Fallback if langmem not available
    create_manage_memory_tool = None
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
                f.write(f"[ManageMemoryTool] {message}\n")
                f.flush()
        except:
            pass
    logger.info(f"[TOOL] {message}")


class ManageMemoryTool(BaseTool):
    """Memory management tool that wraps LangMem management functionality."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        _debug_log("🔧 Initializing ManageMemoryTool...")
        
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        
        # Use shared memory store manager
        from .base_memory_tool import MemoryStoreManager
        self.store_manager = MemoryStoreManager
        
        # Initialize LangMem components
        self.langmem_manage = None
        self._init_langmem()
        
        _debug_log(f"✅ ManageMemoryTool initialized (LangMem available: {self.langmem_manage is not None})")

    def _init_langmem(self):
        """Initialize LangMem components."""
        if create_manage_memory_tool is None or InMemoryStore is None:
            _debug_log("❌ LangMem imports not available (langmem or langgraph missing)")
            logger.warning("LangMem not available, using mock implementation")
            return
            
        try:
            # Create simple text-based store without embeddings
            # This uses exact string matching which is sufficient for our use case
            self.store = InMemoryStore()
            logger.info("Using text-based InMemoryStore (no embeddings required)")
            
            # Create LangMem management tool
            self.langmem_manage = create_manage_memory_tool(
                namespace=("memories",),
                store=self.store
            )
            logger.info("LangMem management tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangMem management tool: {e}")
            self.langmem_manage = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory management."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="manage_memory",
                description="Create, update, or manage memories in the memory database",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "operation": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Type of operation: create, update, or analyze",
                            enum=["create", "update", "analyze"]
                        ),
                        "content": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Memory content to store or update"
                        ),
                        "memory_type": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Type of memory: episodic, semantic, or procedural",
                            enum=["episodic", "semantic", "procedural"]
                        ),
                        "metadata": OpenAIFunctionPropertySchema(
                            type="object",
                            description="Additional metadata for the memory (JSON object)"
                        ),
                        "source": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Reference to the original information source"
                        ),
                        "memory_id": OpenAIFunctionPropertySchema(
                            type="string",
                            description="ID of existing memory to update (required for update operation)"
                        )
                    },
                    required=["operation", "content"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a manage tool instance with initial memory loading."""
        from uuid import uuid4
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize memory store with initial memories if provided
        initial_memories = kwargs.get('initial_memories', [])
        namespace = kwargs.get('namespace', instance_id)
        
        if initial_memories:
            self.store_manager.init_store_with_memories(namespace, initial_memories)
            return instance_id, ToolResponse(text=f"Memory management tool initialized with {len(initial_memories)} memories in namespace '{namespace}'")
        else:
            self.store_manager.get_or_create_store(namespace)  # Ensure store exists
            return instance_id, ToolResponse(text=f"Memory management tool initialized with empty store in namespace '{namespace}'")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory management operation."""
        _debug_log(f"🛠️  ManageMemoryTool.execute called with operation: {parameters.get('operation', 'N/A')}")
        try:
            # Get namespace from kwargs
            namespace = kwargs.get("namespace", instance_id)
            
            operation = parameters.get("operation", "create")
            content = parameters.get("content", "")
            memory_type = parameters.get("memory_type", "episodic")
            metadata = parameters.get("metadata", {})
            source = parameters.get("source", "")
            memory_id = parameters.get("memory_id", None)

            if not content:
                return ToolResponse(text="Error: Content is required for memory management"), 0.0, {}

            # Get shared store for this namespace
            store = self.store_manager.get_or_create_store(namespace)

            # If LangMem is available, use it
            if self.langmem_manage is not None:
                try:
                    if operation == "create":
                        # Create new memory
                        result = await self.langmem_manage.ainvoke({
                            "action": "store",
                            "content": content,
                            "metadata": {**metadata, "type": memory_type, "source": source}
                        })
                        
                        return ToolResponse(
                            text=f"Successfully created {memory_type} memory: {content[:100]}..."
                        ), 0.1, {"operation": operation, "memory_type": memory_type}
                    
                    elif operation == "update":
                        if not memory_id:
                            return ToolResponse(text="Error: memory_id required for update operation"), 0.0, {}
                        
                        # Update existing memory
                        result = await self.langmem_manage.ainvoke({
                            "action": "update", 
                            "memory_id": memory_id,
                            "content": content,
                            "metadata": {**metadata, "type": memory_type, "source": source}
                        })
                        
                        return ToolResponse(
                            text=f"Successfully updated memory {memory_id}: {content[:100]}..."
                        ), 0.1, {"operation": operation, "memory_id": memory_id}
                    
                    elif operation == "analyze":
                        # Analyze and potentially store memory
                        result = await self.langmem_manage.ainvoke({
                            "action": "analyze_and_store",
                            "content": content,
                            "metadata": {**metadata, "type": memory_type, "source": source}
                        })
                        
                        return ToolResponse(
                            text=f"Analyzed and processed content: {content[:100]}..."
                        ), 0.1, {"operation": operation, "analysis": "completed"}
                    
                    else:
                        return ToolResponse(text=f"Error: Unknown operation '{operation}'"), 0.0, {}
                        
                except Exception as e:
                    logger.error(f"LangMem management failed: {e}")
                    return ToolResponse(text=f"Memory management failed: {str(e)}"), 0.0, {}
            else:
                # Mock implementation when LangMem not available
                mock_memory_id = f"mock_{hash(content) % 10000}"
                
                if operation == "create":
                    result_text = f"Mock created {memory_type} memory (ID: {mock_memory_id}): {content[:100]}..."
                elif operation == "update":
                    result_text = f"Mock updated memory {memory_id or mock_memory_id}: {content[:100]}..."
                elif operation == "analyze":
                    result_text = f"Mock analyzed content: {content[:100]}..."
                else:
                    result_text = f"Mock {operation} operation completed"
                
                return ToolResponse(text=result_text), 0.1, {
                    "operation": operation,
                    "memory_id": memory_id or mock_memory_id,
                    "memory_type": memory_type
                }

        except Exception as e:
            logger.error(f"Memory management execution failed: {e}")
            return ToolResponse(text=f"Memory management failed: {str(e)}"), 0.0, {}