"""Memory management tool - wraps LangMem management functionality in verl BaseTool interface."""

import logging
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


class ManageMemoryTool(BaseTool):
    """Memory management tool that wraps LangMem management functionality."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        
        # Use shared memory store manager
        from .base_memory_tool import MemoryStoreManager
        self.store_manager = MemoryStoreManager
        
        # Initialize LangMem components
        self.langmem_manage = None
        self._init_langmem()

    def _init_langmem(self):
        """Initialize LangMem components."""
        if create_manage_memory_tool is None or InMemoryStore is None:
            logger.warning("LangMem not available, using mock implementation")
            return
            
        try:
            # Initialize memory store with embedding configuration
            self.store = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
            
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

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory management operation."""
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