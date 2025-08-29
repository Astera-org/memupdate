"""Memory split tool - decomposes a memory into multiple parts."""

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


class SplitMemoryTool(BaseTool):
    """Memory split tool for decomposing complex memories into smaller parts."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        self.split_memories = {}  # Track split memories per instance

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory splitting."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="split_memory",
                description="Split a complex memory into multiple smaller, focused memories",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "memory_id": OpenAIFunctionPropertySchema(
                            type="string",
                            description="ID of the memory to split"
                        ),
                        "split_criteria": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Criteria for splitting the memory",
                            enum=["temporal", "thematic", "speaker"]
                        ),
                        "max_parts": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Maximum number of parts to split into (default: 3)"
                        )
                    },
                    required=["memory_id", "split_criteria"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a split tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self.split_memories[instance_id] = []
        return instance_id, ToolResponse(text="Memory split tool initialized")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory split operation."""
        try:
            memory_id = parameters.get("memory_id", "")
            split_criteria = parameters.get("split_criteria", "thematic")
            max_parts = parameters.get("max_parts", 3)

            if not memory_id:
                return ToolResponse(text="Error: memory_id is required for splitting"), 0.0, {}
            
            if max_parts < 2 or max_parts > 5:
                return ToolResponse(text="Error: max_parts must be between 2 and 5"), 0.0, {}

            # Mock memory retrieval
            mock_memory_content = f"Complex memory content for {memory_id} containing multiple aspects, timeframes, and speakers."

            # Generate split memory IDs
            new_memory_ids = [f"{memory_id}_part_{i+1}" for i in range(max_parts)]

            # Apply split criteria
            if split_criteria == "temporal":
                split_parts = [
                    f"Temporal part {i+1}: Events from time period {i+1} - {mock_memory_content[:30]}..."
                    for i in range(max_parts)
                ]
            elif split_criteria == "thematic":
                themes = ["emotions", "actions", "locations", "relationships", "outcomes"]
                split_parts = [
                    f"Thematic part {i+1} ({themes[i % len(themes)]}): {mock_memory_content[:40]}..."
                    for i in range(max_parts)
                ]
            elif split_criteria == "speaker":
                speakers = ["Speaker A", "Speaker B", "Speaker C", "Narrator", "Observer"]
                split_parts = [
                    f"Content from {speakers[i % len(speakers)]}: {mock_memory_content[:35]}..."
                    for i in range(max_parts)
                ]
            else:
                split_parts = [
                    f"Split part {i+1}: {mock_memory_content[:40]}..."
                    for i in range(max_parts)
                ]

            # Track the split operation
            if instance_id not in self.split_memories:
                self.split_memories[instance_id] = []
            
            self.split_memories[instance_id].append({
                "original_memory_id": memory_id,
                "new_memory_ids": new_memory_ids,
                "split_criteria": split_criteria,
                "parts_created": len(split_parts)
            })

            result_text = f"Successfully split memory {memory_id} into {len(split_parts)} parts using {split_criteria} criteria:\n\n"
            for i, (new_id, part) in enumerate(zip(new_memory_ids, split_parts), 1):
                result_text += f"{i}. [{new_id}] {part}\n"

            return ToolResponse(text=result_text), 0.1, {
                "original_memory_id": memory_id,
                "new_memory_ids": new_memory_ids,
                "split_criteria": split_criteria,
                "parts_created": len(split_parts)
            }

        except Exception as e:
            logger.error(f"Memory split execution failed: {e}")
            return ToolResponse(text=f"Memory split failed: {str(e)}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the split tool instance."""
        if instance_id in self.split_memories:
            del self.split_memories[instance_id]