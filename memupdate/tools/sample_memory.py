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

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory sampling operation."""
        try:
            k = parameters.get("k", 3)
            memory_type = parameters.get("memory_type", None)
            strategy = parameters.get("strategy", "random")

            if k <= 0:
                return ToolResponse(text="Error: k must be a positive integer"), 0.0, {}

            # Mock implementation - would integrate with actual memory database
            mock_memories = [
                {"id": f"mem_{i}", "content": f"Mock memory {i}", "type": random.choice(["episodic", "semantic", "procedural"])}
                for i in range(1, 11)  # Mock 10 memories
            ]

            # Filter by memory type if specified
            if memory_type:
                mock_memories = [m for m in mock_memories if m["type"] == memory_type]

            # Apply sampling strategy
            if strategy == "random":
                sampled = random.sample(mock_memories, min(k, len(mock_memories)))
            elif strategy == "diverse":
                # Mock diverse sampling - would use embeddings in real implementation
                sampled = mock_memories[:min(k, len(mock_memories))]
            elif strategy == "recent":
                # Mock recent sampling - would use timestamps in real implementation  
                sampled = mock_memories[-min(k, len(mock_memories)):]
            else:
                sampled = random.sample(mock_memories, min(k, len(mock_memories)))

            result_text = f"Sampled {len(sampled)} memories using {strategy} strategy:\n"
            for i, mem in enumerate(sampled, 1):
                result_text += f"{i}. [{mem['type']}] {mem['content']} (ID: {mem['id']})\n"

            return ToolResponse(text=result_text), 0.05, {
                "sampled_count": len(sampled),
                "strategy": strategy,
                "memory_type_filter": memory_type,
                "requested_k": k
            }

        except Exception as e:
            logger.error(f"Memory sampling execution failed: {e}")
            return ToolResponse(text=f"Memory sampling failed: {str(e)}"), 0.0, {}