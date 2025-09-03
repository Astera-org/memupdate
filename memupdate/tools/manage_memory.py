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
        
        print(f"🔧 MEMUPDATE DEBUG: ManageMemoryTool.create called with namespace='{namespace}', instance_id='{instance_id}'")
        
        # Check if there's a create_kwargs that contains the actual namespace and initial_memories
        create_kwargs = kwargs.get('create_kwargs', {})
        if 'namespace' in create_kwargs:
            actual_namespace = create_kwargs['namespace']
            print(f"🔧 MEMUPDATE DEBUG: Found actual namespace in create_kwargs: '{actual_namespace}'")
            namespace = actual_namespace
            
        # 🔧 CRITICAL FIX: Also check for initial_memories in create_kwargs
        if 'initial_memories' in create_kwargs:
            actual_initial_memories = create_kwargs['initial_memories']
            print(f"🔧 MEMUPDATE DEBUG: Found {len(actual_initial_memories)} initial_memories in create_kwargs")
            initial_memories = actual_initial_memories
        
        print(f"🔧 MEMUPDATE DEBUG: Final values - namespace='{namespace}', initial_memories count={len(initial_memories)}")
        
        # 🔧 CRITICAL FIX: Register the mapping from instance_id to intended namespace
        if namespace != instance_id:
            self.store_manager.register_instance_namespace(instance_id, namespace)
        
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
            
            _debug_log(f"📝 MEMUPDATE DEBUG: ManageMemoryTool.execute called with namespace='{namespace}', operation='{operation}'")

            if not content:
                return ToolResponse(text="Error: Content is required for memory management"), 0.0, {}

            # Get shared store for this namespace
            store = self.store_manager.get_or_create_store(namespace)

            # 🔧 CRITICAL FIX: Create LangMem manage tool with shared store per namespace
            if create_manage_memory_tool is not None and InMemoryStore is not None:
                try:
                    # Create LangMem manage tool with the shared store for this namespace
                    langmem_manage = create_manage_memory_tool(
                        namespace=("memories",),
                        store=store  # Use the shared store for this namespace
                    )
                    if operation == "create":
                        # Create new memory
                        result = await langmem_manage.ainvoke({
                            "action": "create",
                            "content": content,
                            "metadata": {**metadata, "type": memory_type, "source": source}
                        })
                        
                        # Memory updated in LangMem store
                        
                        return ToolResponse(
                            text=f"Successfully created {memory_type} memory: {content[:100]}..."
                        ), 0.1, {"operation": operation, "memory_type": memory_type}
                    
                    elif operation == "update":
                        if not memory_id:
                            return ToolResponse(text="Error: memory_id required for update operation"), 0.0, {}
                        
                        # 🔧 FIX: LangMem expects UUID format for "id" field
                        # Convert human-readable ID to UUID if needed
                        import uuid
                        try:
                            # Try to parse as UUID - if it fails, generate a UUID from the string
                            uuid.UUID(memory_id)
                            langmem_id = memory_id
                        except ValueError:
                            # Generate deterministic UUID from string
                            langmem_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))
                            print(f"🔧 Converted memory_id '{memory_id}' to UUID '{langmem_id}'")
                        
                        # Update existing memory
                        result = await langmem_manage.ainvoke({
                            "action": "update", 
                            "id": langmem_id,  # Use UUID-format ID
                            "content": content,
                            "metadata": {**metadata, "type": memory_type, "source": source}
                        })
                        
                        # Memory updated in LangMem store
                        
                        return ToolResponse(
                            text=f"Successfully updated memory {memory_id}: {content[:100]}..."
                        ), 0.1, {"operation": operation, "memory_id": memory_id}
                    
                    elif operation == "analyze":
                        # Analyze and potentially create memory
                        result = await langmem_manage.ainvoke({
                            "action": "create",
                            "content": content,
                            "metadata": {**metadata, "type": memory_type, "source": source}
                        })
                        
                        # Memory updated in LangMem store
                        
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

    async def release(self, instance_id: str, **kwargs):
        """Release tool instance but preserve memory state for reward computation."""
        # 🔧 CRITICAL: Don't clear namespace here - memory state needed for reward computation
        # MemoryStoreManager persists across tool instances using class-level storage
        namespace = kwargs.get("namespace", instance_id)
        
        # 🔧 CRITICAL FIX: Use mapped namespace if available
        namespace = self.store_manager.get_namespace_for_instance(namespace)
        
        _debug_log(f"🧹 ManageMemoryTool.release called for namespace '{namespace}' (memory preserved)")
        
        # Just log current memory state for debugging
        current_memories = self.store_manager.get_current_memories(namespace)
        _debug_log(f"📊 Memory state at release: {len(current_memories)} memories in '{namespace}'")
        
        # Return success (no actual cleanup needed since MemoryStoreManager handles persistence)
        return f"Released ManageMemoryTool instance for namespace '{namespace}'"

