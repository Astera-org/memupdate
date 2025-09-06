"""Memory management tool - wraps LangMem management functionality in verl BaseTool interface."""

import logging
import os
from typing import Any, Optional

print("🔍 ManageMemoryTool: Attempting langmem imports...")
try:
    from langmem import create_manage_memory_tool
    print("✅ ManageMemoryTool: create_manage_memory_tool imported successfully")
    from langgraph.store.memory import InMemoryStore
    print("✅ ManageMemoryTool: InMemoryStore imported successfully")
except ImportError as e:
    print(f"❌ ManageMemoryTool: ImportError during langmem imports: {e}")
    # Fallback if langmem not available
    create_manage_memory_tool = None
    InMemoryStore = None
except Exception as e:
    print(f"❌ ManageMemoryTool: Other error during langmem imports: {type(e).__name__}: {e}")
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
    print(f"[TOOL] {message}")


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
        
        # Store sample_id per instance for execution-time initialization
        self._instance_sample_ids = {}  # instance_id -> sample_id
        
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
            # 🔧 CRITICAL FIX: Don't create separate store - use shared MemoryStoreManager
            # Each tool creating its own InMemoryStore() causes isolation!
            # We'll create LangMem tools with shared stores during execute()
            
            # 🔧 FIX: Set langmem_manage to indicate LangMem is available
            self.langmem_manage = "available"  # Placeholder to indicate LangMem is working
            
            print("✅ ManageMemoryTool will use shared stores per namespace")
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
                            description="Type of operation: create or update",
                            enum=["create", "update"]
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
        """Create a manage tool instance."""
        from uuid import uuid4
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Extract namespace from create_kwargs
        create_kwargs = kwargs.get('create_kwargs', {})
        namespace = create_kwargs.get('namespace', instance_id)
        sample_id = create_kwargs.get('sample_id')
        
        # Store sample_id for this instance (needed during execute)
        if sample_id:
            self._instance_sample_ids[instance_id] = sample_id
        
        # Register the mapping from instance_id to intended namespace if different
        if namespace != instance_id:
            self.store_manager.register_instance_namespace(instance_id, namespace)
        
        # Tool creation is now passive - actual memory initialization happens during execute()
        return instance_id, ToolResponse(text=f"Memory management tool ready for namespace '{namespace}'")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory management operation."""
        _debug_log(f"🛠️  ManageMemoryTool.execute called with operation: {parameters.get('operation', 'N/A')}")
        try:
            print(f"🛠️  MEMUPDATE DEBUG: ManageMemoryTool.execute kwargs: {kwargs}")
            
            # Get namespace from kwargs
            namespace = kwargs.get("namespace", instance_id)
            
            # Check if there's an execute_kwargs that contains the actual namespace
            execute_kwargs = kwargs.get('execute_kwargs', {})
            if 'namespace' in execute_kwargs:
                actual_namespace = execute_kwargs['namespace']
                print(f"🛠️  MEMUPDATE DEBUG: Found actual namespace in execute_kwargs: '{actual_namespace}'")
                namespace = actual_namespace
            
            # 🔧 CRITICAL FIX: Use mapped namespace if available
            namespace = self.store_manager.get_namespace_for_instance(namespace)
            operation = parameters.get("operation", "create")
            content = parameters.get("content", "")
            memory_type = parameters.get("memory_type", "episodic")
            metadata = parameters.get("metadata", {})
            source = parameters.get("source", "")
            memory_id = parameters.get("memory_id", None)
            
            # 🔍 DEBUG: Log the operation being performed
            print(f"🔧 ManageMemoryTool executing '{operation}' operation with content: {content[:100]}...")
            
            _debug_log(f"📝 MEMUPDATE DEBUG: ManageMemoryTool.execute called with namespace='{namespace}', operation='{operation}'")

            if not content:
                return ToolResponse(text="Error: Content is required for memory management"), 0.0, {}

            # 🔧 CRITICAL FIX: Use Ray Actor for all store operations to avoid serialization issues
            # Getting local store copies via ray.get() creates separate instances!
            
            if operation == "create":
                # print(f"🛠️  DEBUG: About to create memory with content: '{content[:100]}...'")
                # print(f"🛠️  DEBUG: Memory type: {memory_type}, metadata: {metadata}")
                
                # Use Ray Actor method directly
                result = self.store_manager.create_memory_via_actor(namespace, {
                    "action": "create",
                    "content": content,
                    "metadata": {**metadata, "type": memory_type, "source": source}
                })
                
                if result["success"]:
                    # print(f"🛠️  DEBUG: Ray Actor create result: {result['result']}")
                    # print(f"🛠️  DEBUG: Store now contains {result['new_count']} memories after create operation in namespace '{namespace}'")
                    return ToolResponse(
                        text=f"Successfully created {memory_type} memory: {content[:100]}..."
                    ), 0.1, {"operation": operation, "memory_type": memory_type}
                else:
                    return ToolResponse(text=f"Failed to create memory: {result['result']}"), 0.0, {}
                    
            elif operation == "update":
                if not memory_id:
                    return ToolResponse(text="Error: memory_id required for update operation"), 0.0, {}
                    
                # Convert human-readable ID to UUID if needed
                import uuid
                try:
                    uuid.UUID(memory_id)
                    langmem_id = memory_id
                except ValueError:
                    langmem_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))
                    print(f"🔧 Converted memory_id '{memory_id}' to UUID '{langmem_id}'")
                
                # Use Ray Actor method directly  
                result = self.store_manager.update_memory_via_actor(namespace, {
                    "action": "update", 
                    "id": langmem_id,
                    "content": content,
                    "metadata": {**metadata, "type": memory_type, "source": source}
                })
                
                if result["success"]:
                    # print(f"🛠️  DEBUG: Ray Actor update result: {result['result']}")
                    # print(f"🛠️  DEBUG: Store now contains {result['new_count']} memories after update operation in namespace '{namespace}'")
                    return ToolResponse(
                        text=f"Successfully updated memory {memory_id}: {content[:100]}..."
                    ), 0.1, {"operation": operation, "memory_id": memory_id}
                else:
                    return ToolResponse(text=f"Failed to update memory: {result['result']}"), 0.0, {}
            else:
                # 🚫 Unknown operation - should only be create or update
                print(f"❌ ERROR: Unknown operation '{operation}' - valid operations are: create or update")
                return ToolResponse(text=f"Error: Unknown operation '{operation}'. Valid operations: create or update"), 0.0, {}

        except Exception as e:
            logger.error(f"Memory management execution failed: {e}")
            return ToolResponse(text=f"Memory management failed: {str(e)}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        """Release tool instance but preserve memory state for reward computation."""
        # 🔧 CRITICAL: Don't clear namespace here - memory state needed for reward computation
        # MemoryStoreManager persists across tool instances using class-level storage
        namespace = kwargs.get("namespace", instance_id)
        
        # 🔧 CRITICAL FIX: Use mapped namespace if available
        namespace = self.store_manager.get_namespace_for_instance(namespace)
        
        _debug_log(f"🧹 ManageMemoryTool.release called for namespace '{namespace}' (memory preserved)")
        
        
        # Return success (no actual cleanup needed since MemoryStoreManager handles persistence)
        return f"Released ManageMemoryTool instance for namespace '{namespace}'"

