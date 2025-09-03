"""Memory sampling tool - samples memories from the database for analysis."""

import logging
import random
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionToolSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    ToolResponse,
)

logger = logging.getLogger(__name__)


class SampleMemoryTool(BaseTool):
    """Memory sampling tool for retrieving random or diverse memory samples."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        
        # Use shared memory store manager
        from .base_memory_tool import MemoryStoreManager
        self.store_manager = MemoryStoreManager

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory sampling."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="sample_memory",
                description="Sample memories from the database using various strategies",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "k": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Number of memories to sample (default: 3)"
                        ),
                        "memory_type": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Filter by memory type",
                            enum=["episodic", "semantic", "procedural"]
                        ),
                        "strategy": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Sampling strategy to use",
                            enum=["random", "diverse", "recent"]
                        )
                    },
                    required=["k"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a sample tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize memory store with initial memories if provided
        initial_memories = kwargs.get('initial_memories', [])
        namespace = kwargs.get('namespace', instance_id)
        
        print(f"📊 MEMUPDATE DEBUG: SampleMemoryTool.create called with namespace='{namespace}', instance_id='{instance_id}'")
        
        # Check if there's a create_kwargs that contains the actual namespace
        create_kwargs = kwargs.get('create_kwargs', {})
        if 'namespace' in create_kwargs:
            actual_namespace = create_kwargs['namespace']
            print(f"📊 MEMUPDATE DEBUG: Found actual namespace in create_kwargs: '{actual_namespace}'")
            namespace = actual_namespace
        
        # 🔧 CRITICAL FIX: Register the mapping from instance_id to intended namespace
        if namespace != instance_id:
            self.store_manager.register_instance_namespace(instance_id, namespace)
        
        if initial_memories:
            self.store_manager.init_store_with_memories(namespace, initial_memories)
            return instance_id, ToolResponse(text=f"Memory sample tool initialized with {len(initial_memories)} memories in namespace '{namespace}'")
        else:
            self.store_manager.get_or_create_store(namespace)  # Ensure store exists
            return instance_id, ToolResponse(text=f"Memory sample tool initialized with empty store in namespace '{namespace}'")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory sampling operation."""
        try:
            k = parameters.get("k", 3)
            memory_type = parameters.get("memory_type", None)
            strategy = parameters.get("strategy", "random")

            if k <= 0:
                return ToolResponse(text="Error: k must be a positive integer"), 0.0, {}

            # Get namespace from kwargs
            namespace = kwargs.get("namespace", instance_id)
            
            # 🔧 CRITICAL FIX: Use mapped namespace if available
            namespace = self.store_manager.get_namespace_for_instance(namespace)
            
            # Get actual memories from the store manager
            current_memories = self.store_manager.get_current_memories(namespace)
            
            # If no memories in cache, return empty results
            if not current_memories:
                return ToolResponse(text="No memories found in the database"), 0.0, {
                    "sampled_count": 0,
                    "strategy": strategy,
                    "memory_type_filter": memory_type,
                    "requested_k": k
                }

            # Filter by memory type if specified
            filtered_memories = current_memories
            if memory_type:
                filtered_memories = [m for m in current_memories 
                                   if m.get("metadata", {}).get("type") == memory_type]

            # Apply sampling strategy
            if strategy == "random":
                sampled = random.sample(filtered_memories, min(k, len(filtered_memories)))
            elif strategy == "diverse":
                # Simple diverse sampling - take evenly spaced memories
                step = max(1, len(filtered_memories) // k)
                sampled = filtered_memories[::step][:k]
            elif strategy == "recent":
                # Take most recent memories (assuming order represents recency)
                sampled = filtered_memories[-min(k, len(filtered_memories)):]
            else:
                sampled = random.sample(filtered_memories, min(k, len(filtered_memories)))

            # Format results with actual content
            result_text = f"Sampled {len(sampled)} memories using {strategy} strategy:\n"
            for i, mem in enumerate(sampled, 1):
                mem_type = mem.get("metadata", {}).get("type", "unknown")
                content = mem.get("content", "No content")[:200]  # Truncate long content
                mem_id = mem.get("id", f"mem_{i}")
                result_text += f"{i}. [{mem_type}] {content} (ID: {mem_id})\n"

            return ToolResponse(text=result_text), 0.05, {
                "sampled_count": len(sampled),
                "strategy": strategy,
                "memory_type_filter": memory_type,
                "requested_k": k
            }

        except Exception as e:
            logger.error(f"Memory sampling execution failed: {e}")
            return ToolResponse(text=f"Memory sampling failed: {str(e)}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        """Release tool instance but preserve memory state for reward computation."""
        namespace = kwargs.get("namespace", instance_id)
        
        # 🔧 CRITICAL FIX: Use mapped namespace if available
        namespace = self.store_manager.get_namespace_for_instance(namespace)
        
        current_memories = self.store_manager.get_current_memories(namespace)
        print(f"🧹 SampleMemoryTool.release: {len(current_memories)} memories in '{namespace}' (preserved)")
        return f"Released SampleMemoryTool instance for namespace '{namespace}'"