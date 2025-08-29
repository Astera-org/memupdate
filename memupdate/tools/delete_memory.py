"""Memory deletion tool - deletes memories from the memory database."""

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


class DeleteMemoryTool(BaseTool):
    """Memory deletion tool for removing specific memories from the database."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        self.deleted_memories = {}  # Track deleted memories per instance

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory deletion."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="delete_memory",
                description="Delete specific memories from the memory database",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "memory_id": OpenAIFunctionPropertySchema(
                            type="string",
                            description="ID of the memory to delete"
                        ),
                        "reason": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Reason for deleting this memory (optional)"
                        )
                    },
                    required=["memory_id"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a deletion tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self.deleted_memories[instance_id] = []
        return instance_id, ToolResponse(text="Memory deletion tool initialized")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory deletion operation."""
        try:
            memory_id = parameters.get("memory_id", "")
            reason = parameters.get("reason", "No reason provided")

            if not memory_id:
                return ToolResponse(text="Error: memory_id is required for deletion"), 0.0, {}

            # Track deleted memory
            if instance_id not in self.deleted_memories:
                self.deleted_memories[instance_id] = []
            
            self.deleted_memories[instance_id].append({
                "memory_id": memory_id,
                "reason": reason,
                "deleted_at": "mock_timestamp"
            })

            return ToolResponse(
                text=f"Successfully deleted memory {memory_id}. Reason: {reason}"
            ), 0.05, {
                "deleted_memory_id": memory_id,
                "reason": reason,
                "total_deleted": len(self.deleted_memories[instance_id])
            }

        except Exception as e:
            logger.error(f"Memory deletion execution failed: {e}")
            return ToolResponse(text=f"Memory deletion failed: {str(e)}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the deletion tool instance."""
        if instance_id in self.deleted_memories:
            del self.deleted_memories[instance_id]