"""Memory reward manager for MemUpdate RL training."""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

import torch

try:
    from verl.workers.reward_manager.abstract import AbstractRewardManager
    from verl.protocol import DataProto
    from verl.workers.reward_manager import register
    VERL_AVAILABLE = True
except ImportError:
    # Fallback for when verl is not available
    class AbstractRewardManager:
        pass
    class DataProto:
        pass
    register = lambda name: lambda cls: cls
    VERL_AVAILABLE = False

logger = logging.getLogger(__name__)


@register("memory_rag")
class MemoryRewardManager(AbstractRewardManager):
    """Manages reward computation for memory update operations."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        # Store verl parameters
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        
        # MemUpdate-specific config from kwargs
        self.max_total_memories = kwargs.get("max_total_memories", 100)
        logger.info(f"Initialized MemoryRewardManager with max_memories={self.max_total_memories}")

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Main entry point for reward computation."""
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            try:
                # Extract memory states from extra_info
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                
                # Get memory states (set by tools during episode)
                initial_memories = extra_info.get("initial_memories", [])
                final_memories = extra_info.get("final_memories", initial_memories)
                target_question = extra_info.get("target_question", "")
                target_answer = extra_info.get("target_answer", "")
                
                # Compute reward for this episode
                episode_reward = self.compute_single_reward(
                    initial_memories, 
                    final_memories,
                    target_question,
                    target_answer
                )
                
                # Decode response for validation length
                response_ids = data_item.batch["responses"]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                
                # Set reward at the end of the valid response
                reward_tensor[i, valid_response_length - 1] = episode_reward
                
                # Store extra info
                reward_extra_info["memory_reward"].append(episode_reward)
                reward_extra_info["initial_memory_count"].append(len(initial_memories))
                reward_extra_info["final_memory_count"].append(len(final_memories))
                
                # Debug printing
                if i < self.num_examine:
                    prompt_ids = data_item.batch["prompts"]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                    valid_response_ids = response_ids[:valid_response_length]
                    
                    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    
                    logger.info(f"[MemUpdate Reward {i}]")
                    logger.info(f"[question] {target_question}")
                    logger.info(f"[target_answer] {target_answer}")
                    logger.info(f"[initial_memories] {len(initial_memories)}")
                    logger.info(f"[final_memories] {len(final_memories)}")
                    logger.info(f"[reward] {episode_reward:.3f}")
                
            except Exception as e:
                logger.error(f"Error computing reward for batch item {i}: {e}")
                # Small penalty for errors
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                reward_tensor[i, valid_response_length - 1] = -0.1
                reward_extra_info["memory_reward"].append(-0.1)
                reward_extra_info["error"].append(str(e))
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def compute_single_reward(
        self,
        memory_old: List[Dict],
        memory_new: List[Dict], 
        target_question: str,
        target_answer: str,
    ) -> float:
        """
        Compute reward for a single memory update operation.
        
        Following original design:
        1. Evaluate QA performance with old memory
        2. Evaluate QA performance with new memory  
        3. Reward = performance_delta × memory_efficiency
        """
        try:
            # 1. Compute QA performance difference
            performance_old = self.evaluate_single_qa(memory_old, target_question, target_answer)
            performance_new = self.evaluate_single_qa(memory_new, target_question, target_answer)
            performance_delta = performance_new - performance_old
            
            # 2. Memory efficiency factor
            memory_efficiency = self._compute_memory_efficiency(memory_old, memory_new)
            
            # 3. Final reward computation
            reward = performance_delta * memory_efficiency
            
            return reward
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            return -0.1  # Small penalty for errors

    def evaluate_single_qa(self, memory_db: List[Dict], question: str, answer: str) -> float:
        """
        Evaluate QA performance for a single question using RAG + context overlap.
        
        Returns score between 0 and 1.
        """
        try:
            # 1. Retrieve relevant memories (RAG)
            context_memories = self._rag_retrieve(memory_db, question, top_k=5)
            
            # 2. Build context string
            if not context_memories:
                context = "No relevant information found."
            else:
                context = "\n".join([
                    f"Memory: {mem.get('content', '')}" 
                    for mem in context_memories
                ])
            
            # 3. Score based on context-answer overlap (proxy for model performance)
            # Higher score = better memory retrieval supports the correct answer
            target_tokens = set(str(answer).lower().split())
            context_tokens = set(context.lower().split())
            
            if not target_tokens:
                return 0.5  # Neutral score for empty target
                
            overlap = len(target_tokens.intersection(context_tokens))
            support_score = overlap / len(target_tokens)  # 0 to 1
            
            # Convert to 0-1 probability score (higher = better memory)
            return max(0.1, min(1.0, support_score))
            
        except Exception as e:
            logger.error(f"QA evaluation failed: {e}")
            return 0.1  # Small positive baseline

    def _rag_retrieve(self, memory_db: List[Dict], question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant memories for the question.
        Simple implementation using keyword matching.
        """
        if not memory_db:
            return []
            
        # Simple keyword-based retrieval (would use embeddings in production)
        question_words = set(question.lower().split())
        
        scored_memories = []
        for memory in memory_db:
            content = memory.get("content", "").lower()
            memory_words = set(content.split())
            
            # Compute simple overlap score
            overlap = len(question_words.intersection(memory_words))
            if overlap > 0:
                score = overlap / max(len(question_words), len(memory_words))
                scored_memories.append((score, memory))
        
        # Sort by relevance and return top-k
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [mem for _, mem in scored_memories[:top_k]]

    def _compute_memory_efficiency(self, memory_old: List[Dict], memory_new: List[Dict]) -> float:
        """
        Compute memory efficiency factor.
        
        Rewards:
        - Keeping memory size reasonable
        - Reducing redundancy
        - Maintaining information density
        """
        old_size = len(memory_old)
        new_size = len(memory_new)
        
        # Size penalty/bonus
        if new_size > self.max_total_memories:
            size_factor = max(0.1, 1.0 - (new_size - self.max_total_memories) / self.max_total_memories)
        else:
            size_factor = 1.0
        
        # Information density bonus (more information per memory unit)
        if old_size > 0:
            total_content_old = sum(len(mem.get("content", "")) for mem in memory_old)
            total_content_new = sum(len(mem.get("content", "")) for mem in memory_new)
            
            density_old = total_content_old / max(old_size, 1)
            density_new = total_content_new / max(new_size, 1)
            
            density_factor = min(1.2, density_new / max(density_old, 1))
        else:
            density_factor = 1.0
            
        # Change magnitude factor (don't reward excessive changes)
        change_ratio = abs(new_size - old_size) / max(old_size, 1)
        change_factor = max(0.8, 1.0 - change_ratio * 0.5)
        
        efficiency = size_factor * density_factor * change_factor
        return max(0.1, min(1.5, efficiency))  # Clamp between 0.1 and 1.5