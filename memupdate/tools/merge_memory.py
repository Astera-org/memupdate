"""Memory merge tool - consolidates multiple memories into one."""

import logging
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


class MergeMemoryTool(BaseTool):
    """Memory merge tool for consolidating multiple related memories."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        self.merged_memories = {}  # Track merged memories per instance

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory merging."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="merge_memory",
                description="Merge multiple related memories into a single consolidated memory",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "memory_ids": OpenAIFunctionPropertySchema(
                            type="array",
                            description="List of 2-5 memory IDs to merge together"
                        ),
                        "strategy": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Merge strategy to use",
                            enum=["summarize", "concatenate", "extract_common"]
                        )
                    },
                    required=["memory_ids", "strategy"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a merge tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self.merged_memories[instance_id] = []
        return instance_id, ToolResponse(text="Memory merge tool initialized")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory merge operation."""
        try:
            memory_ids = parameters.get("memory_ids", [])
            strategy = parameters.get("strategy", "summarize")

            if not memory_ids or len(memory_ids) < 2:
                return ToolResponse(text="Error: At least 2 memory IDs required for merging"), 0.0, {}
            
            if len(memory_ids) > 5:
                return ToolResponse(text="Error: Cannot merge more than 5 memories at once"), 0.0, {}

            # Mock memory retrieval and merging
            mock_memories = {
                mem_id: f"Mock content for memory {mem_id}" for mem_id in memory_ids
            }

            # Generate new merged memory ID
            new_memory_id = f"merged_{hash(''.join(memory_ids)) % 10000}"

            # Apply merge strategy
            if strategy == "summarize":
                merged_content = f"Summary of {len(memory_ids)} memories: " + "; ".join(
                    f"Memory {mid}: {content[:50]}..." for mid, content in mock_memories.items()
                )
            elif strategy == "concatenate":
                merged_content = "\n---\n".join(
                    f"[{mid}] {content}" for mid, content in mock_memories.items()
                )
            elif strategy == "extract_common":
                merged_content = f"Common themes from {len(memory_ids)} memories: " + ", ".join(
                    f"Theme from {mid}" for mid in memory_ids
                )
            else:
                merged_content = f"Merged {len(memory_ids)} memories using {strategy} strategy"

            # Track the merge operation
            if instance_id not in self.merged_memories:
                self.merged_memories[instance_id] = []
            
            self.merged_memories[instance_id].append({
                "new_memory_id": new_memory_id,
                "source_memory_ids": memory_ids,
                "strategy": strategy,
                "merged_content": merged_content[:200] + "..." if len(merged_content) > 200 else merged_content
            })

            return ToolResponse(
                text=f"Successfully merged {len(memory_ids)} memories into new memory {new_memory_id} using {strategy} strategy.\n\nMerged content preview: {merged_content[:150]}..."
            ), 0.15, {
                "new_memory_id": new_memory_id,
                "source_memory_ids": memory_ids,
                "strategy": strategy,
                "merged_count": len(memory_ids)
            }

        except Exception as e:
            logger.error(f"Memory merge execution failed: {e}")
            return ToolResponse(text=f"Memory merge failed: {str(e)}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the merge tool instance."""
        if instance_id in self.merged_memories:
            del self.merged_memories[instance_id]