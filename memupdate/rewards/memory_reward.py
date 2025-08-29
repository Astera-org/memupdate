"""Memory reward manager for MemUpdate RL training."""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    import openai
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    openai = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from verl.workers.reward_manager.abstract import AbstractRewardManager
    from verl.protocol import DataProto
    import torch
    VERL_AVAILABLE = True
except ImportError:
    # Fallback for when verl is not available
    class AbstractRewardManager:
        pass
    class DataProto:
        pass
    torch = None
    VERL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryRewardManager(AbstractRewardManager):
    """Manages reward computation for memory update operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_total_memories = config.get("max_total_memories", 100)
        self.evaluator_model = config.get("evaluator_model", "openai:gpt-4o-mini")
        self.use_llm_judge = config.get("use_llm_judge", True)
        
        # Initialize evaluation model
        self._init_evaluator()

    def _init_evaluator(self):
        """Initialize the evaluation model."""
        if self.evaluator_model.startswith("openai:"):
            if openai is None:
                logger.warning("OpenAI not available, using mock evaluation")
                self.evaluator = None
            else:
                model_name = self.evaluator_model.replace("openai:", "")
                self.evaluator = openai.OpenAI()
                self.model_name = model_name
                logger.info(f"Initialized OpenAI evaluator: {model_name}")
        else:
            logger.warning("Only OpenAI models supported currently, using mock evaluation")
            self.evaluator = None

    async def compute_reward(self, data: DataProto) -> DataProto:
        """
        Compute rewards for a batch of memory update episodes.
        
        IMPORTANT: verl calls this with a DataProto batch, not individual samples.
        """
        if not VERL_AVAILABLE:
            logger.warning("verl not available, using mock reward computation")
            return 0.1
            
        batch_size = len(data)
        rewards = []
        
        for i in range(batch_size):
            try:
                # Extract episode data from verl's DataProto
                extra_info = data.get_nested_field("extra_info")[i]
                
                # Get memory states
                initial_memories = extra_info.get("initial_memories", [])
                final_memories = extra_info.get("final_memories", initial_memories)  # Set by tools during episode
                target_question = extra_info.get("target_question", "")
                target_answer = extra_info.get("target_answer", "")
                
                # Compute reward for this episode
                episode_reward = await self.compute_single_reward(
                    initial_memories, 
                    final_memories,
                    target_question,
                    target_answer
                )
                rewards.append(episode_reward)
            except Exception as e:
                logger.error(f"Error computing reward for batch item {i}: {e}")
                rewards.append(-0.1)  # Small penalty for errors
        
        # Return rewards as DataProto
        if torch is not None:
            return DataProto(rewards=torch.tensor(rewards, dtype=torch.float32))
        else:
            return rewards  # Fallback

    async def compute_single_reward(
        self,
        memory_old: List[Dict],
        memory_new: List[Dict], 
        target_question: str,
        target_answer: str,
        **kwargs
    ) -> float:
        """
        Compute reward for a single memory update operation.
        
        Args:
            memory_old: Original memory database
            memory_new: Updated memory database  
            target_question: Question for this episode
            target_answer: Ground truth answer
            
        Returns:
            Reward score (can be negative)
        """
        try:
            # 1. Compute QA performance difference
            performance_old = await self.evaluate_single_qa(
                memory_old, target_question, target_answer
            )
            performance_new = await self.evaluate_single_qa(
                memory_new, target_question, target_answer
            )
            
            performance_delta = performance_new - performance_old
            
            # 2. Memory efficiency penalty/bonus
            memory_efficiency = self._compute_memory_efficiency(memory_old, memory_new)
            
            # 3. Final reward computation
            reward = performance_delta * memory_efficiency
            
            logger.info(
                f"Reward computation: performance_delta={performance_delta:.3f}, "
                f"memory_efficiency={memory_efficiency:.3f}, final_reward={reward:.3f}"
            )
            
            return reward
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            return -0.1  # Small penalty for errors

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

    async def evaluate_single_qa(
        self, 
        memory_db: List[Dict], 
        question: str, 
        answer: str
    ) -> float:
        """
        Evaluate QA performance for a single question.
        
        Returns score between 0 and 1.
        """
        try:
            # 1. Retrieve relevant memories (RAG)
            context_memories = self._rag_retrieve(memory_db, question, top_k=5)
            
            # 2. Generate answer using retrieved context
            generated_answer = await self._generate_answer(question, context_memories)
            
            # 3. Evaluate answer quality
            score = await self._evaluate_answer(generated_answer, answer)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"QA evaluation failed: {e}")
            return 0.0

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

    async def _generate_answer(self, question: str, context_memories: List[Dict]) -> str:
        """Generate answer using question and retrieved memories."""
        if not context_memories:
            context_text = "No relevant memories found."
        else:
            context_text = "\\n".join([
                f"Memory {i+1}: {mem.get('content', '')}" 
                for i, mem in enumerate(context_memories)
            ])
        
        prompt = f"""Answer the following question based on the provided memory context.

Context from memories:
{context_text}

Question: {question}

Answer:"""

        if self.evaluator is not None:
            try:
                response = await self.evaluator.chat.completions.acreate(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                return "Mock generated answer based on context"
        else:
            # Mock answer generation
            return f"Mock answer for: {question} (based on {len(context_memories)} memories)"

    async def _evaluate_answer(self, generated_answer: str, target_answer: str) -> float:
        """
        Evaluate generated answer against target answer.
        
        Uses combination of exact match, token overlap, and LLM judge.
        """
        # 1. Exact match
        if generated_answer.strip().lower() == target_answer.strip().lower():
            return 1.0
        
        # 2. Token overlap (F1-style)
        gen_tokens = set(generated_answer.lower().split())
        target_tokens = set(str(target_answer).lower().split())
        
        if not gen_tokens and not target_tokens:
            return 1.0
        elif not gen_tokens or not target_tokens:
            return 0.0
        
        intersection = gen_tokens.intersection(target_tokens)
        precision = len(intersection) / len(gen_tokens)
        recall = len(intersection) / len(target_tokens)
        
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        # 3. LLM judge (if available and enabled)
        if self.use_llm_judge and self.evaluator is not None:
            try:
                judge_score = await self._llm_judge(generated_answer, target_answer)
                # Combine F1 and LLM judge scores
                final_score = 0.6 * f1_score + 0.4 * judge_score
            except Exception as e:
                logger.error(f"LLM judge evaluation failed: {e}")
                final_score = f1_score
        else:
            final_score = f1_score
            
        return max(0.0, min(1.0, final_score))

    async def _llm_judge(self, generated_answer: str, target_answer: str) -> float:
        """Use LLM as a judge to evaluate answer quality."""
        prompt = f"""Evaluate how well the generated answer matches the target answer on a scale of 0.0 to 1.0.

Consider:
- Factual correctness
- Completeness of information
- Semantic similarity

Target Answer: {target_answer}
Generated Answer: {generated_answer}

Provide only a numeric score between 0.0 and 1.0:"""

        try:
            response = await self.evaluator.chat.completions.acreate(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
            
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse LLM judge score: {e}")
            return 0.5  # Default neutral score
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return 0.5


def register_reward_manager():
    """Register the reward manager with verl."""
    try:
        from verl.workers.reward_manager import register
        register("memory_rag")(MemoryRewardManager)
        logger.info("Registered MemoryRewardManager with verl")
    except ImportError:
        logger.warning("Could not register reward manager - verl not available")