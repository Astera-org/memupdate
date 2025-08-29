"""LoCoMo dataset preprocessing for verl training format."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

logger = logging.getLogger(__name__)


class LoCoMoProcessor:
    """Processes LoCoMo dataset for MemUpdate training."""

    def __init__(self, locomo_data_path: str = "/data/users/alan/locomo/data/locomo10.json"):
        self.locomo_data_path = Path(locomo_data_path)
        self.data = None
        
    def load_data(self):
        """Load LoCoMo dataset from JSON file."""
        try:
            with open(self.locomo_data_path, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} conversations from {self.locomo_data_path}")
        except FileNotFoundError:
            logger.error(f"LoCoMo data file not found: {self.locomo_data_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LoCoMo JSON data: {e}")
            raise

    def create_train_test_split(self, train_conversations: int = 7, seed: int = 42):
        """Create train/test split from conversations."""
        if not self.data:
            self.load_data()
            
        random.seed(seed)
        conversations = list(range(len(self.data)))
        random.shuffle(conversations)
        
        train_indices = conversations[:train_conversations]
        test_indices = conversations[train_conversations:]
        
        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]
        
        logger.info(f"Split data: {len(train_data)} train conversations, {len(test_data)} test conversations")
        return train_data, test_data

    def extract_qa_pairs(self, conversations: List[Dict]) -> List[Dict]:
        """Extract QA pairs from conversations for training trials."""
        qa_trials = []
        
        for conv in conversations:
            sample_id = conv.get("sample_id", "unknown")
            qa_pairs = conv.get("qa", [])
            conversation_data = conv.get("conversation", {})  # LoCoMo uses "conversation", not "conversations"
            facts = conv.get("observation", {})  # LoCoMo uses "observation", not "facts"
            
            for qa in qa_pairs:
                trial = {
                    "sample_id": sample_id,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "evidence": qa.get("evidence", []),
                    "category": qa.get("category", 0),
                    "conversations": conversation_data,
                    "facts": facts  # This will be the observation data
                }
                qa_trials.append(trial)
                
        logger.info(f"Extracted {len(qa_trials)} QA trials")
        return qa_trials

    def convert_facts_to_memories(self, observations: Dict) -> List[Dict]:
        """Convert conversation observations to initial memory entries."""
        memories = []
        
        # LoCoMo observations are nested under session keys
        for session_key, session_obs in observations.items():
            if isinstance(session_obs, dict):
                for speaker, speaker_facts in session_obs.items():
                    for fact_entry in speaker_facts:
                        if isinstance(fact_entry, list) and len(fact_entry) >= 2:
                            fact_text, evidence = fact_entry[0], fact_entry[1]
                            
                            memory = {
                                "content": fact_text,
                                "speaker": speaker,
                                "evidence": evidence,
                                "session": session_key,
                                "memory_type": "episodic",  # Default type
                                "source": "conversation_observations"
                            }
                            memories.append(memory)
                    
        return memories

    def create_verl_training_data(self, qa_trials: List[Dict]) -> List[Dict]:
        """Convert QA trials to verl training format with full context in messages."""
        training_data = []
        
        for idx, trial in enumerate(qa_trials):
            # Extract initial memories from conversation facts
            initial_memories = self.convert_facts_to_memories(trial["facts"])
            
            # Don't hardcode memory in prompt - let LLM discover via tool calls!
            
            # Create comprehensive system prompt with all context
            system_content = """You are a memory management agent with access to memory tools. Your task is to analyze the current memory database and use the available tools to optimize it for better question answering.

Available tools:
- search_memory: Search and retrieve relevant memories
- manage_memory: Create or update memory entries
- delete_memory: Remove outdated or irrelevant memories  
- sample_memory: Sample diverse memories for analysis
- merge_memory: Consolidate related memories
- split_memory: Break down complex memories

Use these tools strategically to improve the memory database for the target question. 

IMPORTANT: Call functions using this EXACT JSON format:
{
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "search_memory",
        "arguments": {"query": "your search query here", "limit": 5}
      }
    }
  ]
}

For manage_memory:
{
  "tool_calls": [
    {
      "type": "function", 
      "function": {
        "name": "manage_memory",
        "arguments": {"operation": "create", "content": "memory content", "memory_type": "episodic"}
      }
    }
  ]
}

For delete_memory:
{
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "delete_memory", 
        "arguments": {"memory_ids": ["id1", "id2"]}
      }
    }
  ]
}

Always use this exact structure for tool calls."""

            # Create user prompt - LLM must use tools to discover memory state
            user_content = f"""Target question to optimize for: {trial['question']}

Please analyze and update the memory database to ensure this question can be answered correctly. 

IMPORTANT: Start by calling search_memory() to see what information is currently stored in the memory database, then use the other tools as needed to optimize it for answering the target question."""

            # IMPORTANT: verl expects these exact keys in this format
            record = {
                # OpenAI chat format messages - agent sees EVERYTHING here
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                "prompt": f"Target question: {trial['question']}",  # Fallback prompt field
                "data_source": f"locomo-{trial['sample_id']}",
                "extra_info": {
                    "index": idx,
                    "need_tools_kwargs": True,  # CRITICAL: This enables tool usage
                    "tools_kwargs": {
                        # Each tool needs namespace for conversation isolation and initial memory loading
                        "search_memory": {
                            "create_kwargs": {"initial_memories": initial_memories, "namespace": trial['sample_id']},
                            "execute_kwargs": {"namespace": trial['sample_id']},
                            "calc_reward_kwargs": {"namespace": trial['sample_id']}
                        },
                        "manage_memory": {
                            "create_kwargs": {"initial_memories": initial_memories, "namespace": trial['sample_id']},
                            "execute_kwargs": {"namespace": trial['sample_id']},
                            "calc_reward_kwargs": {"namespace": trial['sample_id']}
                        },
                        "delete_memory": {
                            "create_kwargs": {"initial_memories": initial_memories, "namespace": trial['sample_id']},
                            "execute_kwargs": {"namespace": trial['sample_id']}
                        },
                        "sample_memory": {
                            "create_kwargs": {"initial_memories": initial_memories, "namespace": trial['sample_id']},
                            "execute_kwargs": {"namespace": trial['sample_id']}
                        },
                        "merge_memory": {
                            "create_kwargs": {"initial_memories": initial_memories, "namespace": trial['sample_id']},
                            "execute_kwargs": {"namespace": trial['sample_id']}
                        },
                        "split_memory": {
                            "create_kwargs": {"initial_memories": initial_memories, "namespace": trial['sample_id']},
                            "execute_kwargs": {"namespace": trial['sample_id']}
                        }
                    },
                    # Data for reward computation
                    "target_question": trial["question"],
                    "target_answer": trial["answer"],
                    "conversation_id": trial["sample_id"],
                    "initial_memories": initial_memories,
                    "evidence": trial.get("evidence", []),
                    "category": trial.get("category", 0)
                }
            }
            training_data.append(record)
        
        return training_data

    def _create_training_prompt(self, memories: List[Dict], target_question: str, conversation_context: Dict) -> str:
        """Create training prompt for memory update agent."""
        
        # Format initial memories
        memory_text = "Initial Memory Database:\\n"
        for i, mem in enumerate(memories, 1):
            memory_text += f"{i}. [{mem['memory_type']}] {mem['content']} (from {mem['speaker']})\\n"
        
        # Create context about conversation
        context_text = f"\\nConversation Context:\\n"
        context_text += f"Number of sessions: {len([k for k in conversation_context.keys() if k.startswith('session')])//2}\\n"
        
        prompt = f"""You are a memory management agent tasked with optimizing a memory database to better answer questions.

{memory_text}

{context_text}

Target Question: {target_question}

Your goal is to use the available memory tools (search, manage, delete, sample, merge, split) to improve the memory database so that it can better answer the target question. You have a maximum of 30 tool calls to optimize the memory database.

Consider:
1. Are there redundant memories that should be merged?
2. Are there complex memories that should be split into focused parts?
3. Are there missing connections or inferences that should be added?
4. Are there low-quality memories that should be deleted?
5. Would sampling help identify patterns for better organization?

Begin optimizing the memory database now."""
        
        return prompt

    def save_parquet(self, data: List[Dict], output_path: str):
        """Save processed data to parquet format for verl."""
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(data)} examples to {output_path}")

    def save_parquet_files(self, output_dir: str = "data/locomo"):
        """Save training data as parquet files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and split data
        train_conversations, test_conversations = self.create_train_test_split()
        
        # Extract QA pairs
        train_trials = self.extract_qa_pairs(train_conversations)
        test_trials = self.extract_qa_pairs(test_conversations)
        
        # Convert to verl format
        train_data = self.create_verl_training_data(train_trials)
        test_data = self.create_verl_training_data(test_trials)
        
        # Save as parquet - convert complex objects to JSON strings for parquet compatibility
        import json
        
        def serialize_complex_fields(df):
            """Convert complex Python objects to JSON strings for parquet storage."""
            df_copy = df.copy()
            if 'extra_info' in df_copy.columns:
                df_copy['extra_info'] = df_copy['extra_info'].apply(json.dumps)
            if 'messages' in df_copy.columns:
                df_copy['messages'] = df_copy['messages'].apply(json.dumps)
            return df_copy
        
        train_df = serialize_complex_fields(pd.DataFrame(train_data))
        test_df = serialize_complex_fields(pd.DataFrame(test_data))
        
        train_df.to_parquet(f"{output_dir}/train_corrected.parquet", index=False)
        test_df.to_parquet(f"{output_dir}/test_corrected.parquet", index=False)
        
        print(f"✅ Saved {len(train_data)} training samples to {output_dir}/train.parquet")
        print(f"✅ Saved {len(test_data)} test samples to {output_dir}/test.parquet")
        return train_df, test_df

    def process_full_pipeline(self, output_dir: str = "/data/users/alan/memupdate/data/locomo"):
        """Run the complete preprocessing pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use the new save_parquet_files method
        train_df, test_df = self.save_parquet_files(str(output_path))
        
        # Save summary stats
        stats = {
            "total_conversations": 10,  # LoCoMo has 10 conversations
            "train_conversations": 7,
            "test_conversations": 3,
            "train_qa_pairs": len(train_df),
            "test_qa_pairs": len(test_df),
            "total_training_examples": len(train_df),
            "total_test_examples": len(test_df)
        }
        
        with open(output_path / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Preprocessing complete. Stats: {stats}")
        return stats


def main():
    """Run preprocessing from command line."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/locomo", help="Output directory")
    parser.add_argument("--input", default="/data/users/alan/locomo/data/locomo10.json")
    args = parser.parse_args()
    
    processor = LoCoMoProcessor(args.input)
    processor.save_parquet_files(args.output)


if __name__ == "__main__":
    main()