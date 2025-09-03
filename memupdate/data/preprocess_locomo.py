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

    def __init__(self, locomo_data_path: str = "/workspace/locomo/data/locomo10.json"):
        self.locomo_data_path = Path(locomo_data_path)
        self.data = None

    def load_data(self):
        """Load LoCoMo dataset from JSON file."""
        try:
            with open(self.locomo_data_path, "r") as f:
                self.data = json.load(f)
            logger.info(
                f"Loaded {len(self.data)} conversations from {self.locomo_data_path}"
            )
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

        logger.info(
            f"Split data: {len(train_data)} train conversations, {len(test_data)} test conversations"
        )
        return train_data, test_data

    def extract_qa_pairs(self, conversations: List[Dict]) -> List[Dict]:
        """Extract QA pairs from conversations for training trials."""
        qa_trials = []
        import uuid

        for conv in conversations:
            sample_id = conv.get("sample_id", "unknown")
            qa_pairs = conv.get("qa", [])
            conversation_data = conv.get(
                "conversation", {}
            )  # LoCoMo uses "conversation", not "conversations"
            facts = conv.get(
                "observation", {}
            )  # LoCoMo uses "observation", not "facts"

            for qa_idx, qa in enumerate(qa_pairs):
                # 🔧 CRITICAL FIX: Create unique trajectory ID for each QA pair
                # Each QA pair gets its own namespace to prevent memory bank sharing
                trajectory_id = f"{sample_id}-qa{qa_idx}-{str(uuid.uuid4())[:8]}"
                
                trial = {
                    "sample_id": sample_id,  # Original conversation ID
                    "trajectory_id": trajectory_id,  # Unique trajectory ID
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "evidence": qa.get("evidence", []),
                    "category": qa.get("category", 0),
                    "conversations": conversation_data,
                    "facts": facts,  # This will be the observation data
                }
                qa_trials.append(trial)

        logger.info(f"Extracted {len(qa_trials)} QA trials with unique trajectory IDs")
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
                                "source": "conversation_observations",
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

            # Create system prompt (tools will be auto-injected by tokenizer)
            system_content = """You are a memory management agent. Your task is to optimize a memory database to better answer future questions.

IMPORTANT: Because we're in debug mode still, you need to help me call these tools to see what the memory database looks like.
1. search_memory() to discover the current memory state
2. manage_memory() to update the memory database
3. manage_memory() to create a memory
"""

            # Create user prompt
            user_content = f"""Target question to optimize for: {trial["question"]}

Analyze and update the memory database to ensure this question can be answered correctly."""

            # IMPORTANT: verl expects these exact keys in this format
            record = {
                # verl expects 'prompt' field with list of messages (not JSON string!)
                "prompt": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                "data_source": f"locomo-{trial['sample_id']}",
                "agent_name": "tool_agent",  # CRITICAL: This enables tool agent loop instead of single_turn_agent
                "extra_info": {
                    "index": idx,
                    "need_tools_kwargs": True,  # CRITICAL: This enables tool usage
                    "tools_kwargs": {
                        # 🔧 CRITICAL FIX: Use unique trajectory_id for namespace isolation
                        # Each QA pair gets its own memory bank to prevent cross-contamination
                        "search_memory": {
                            "create_kwargs": {
                                "initial_memories": initial_memories,
                                "namespace": trial["trajectory_id"],
                            },
                            "execute_kwargs": {"namespace": trial["trajectory_id"]},
                            "calc_reward_kwargs": {"namespace": trial["trajectory_id"]},
                        },
                        "manage_memory": {
                            "create_kwargs": {
                                "initial_memories": initial_memories,
                                "namespace": trial["trajectory_id"],
                            },
                            "execute_kwargs": {"namespace": trial["trajectory_id"]},
                            "calc_reward_kwargs": {"namespace": trial["trajectory_id"]},
                        },
                        "delete_memory": {
                            "create_kwargs": {
                                "initial_memories": initial_memories,
                                "namespace": trial["trajectory_id"],
                            },
                            "execute_kwargs": {"namespace": trial["trajectory_id"]},
                        },
                        "sample_memory": {
                            "create_kwargs": {
                                "initial_memories": initial_memories,
                                "namespace": trial["trajectory_id"],
                            },
                            "execute_kwargs": {"namespace": trial["trajectory_id"]},
                        },
                        "merge_memory": {
                            "create_kwargs": {
                                "initial_memories": initial_memories,
                                "namespace": trial["trajectory_id"],
                            },
                            "execute_kwargs": {"namespace": trial["trajectory_id"]},
                        },
                        "split_memory": {
                            "create_kwargs": {
                                "initial_memories": initial_memories,
                                "namespace": trial["trajectory_id"],
                            },
                            "execute_kwargs": {"namespace": trial["trajectory_id"]},
                        },
                    },
                    # Data for reward computation
                    "target_question": trial["question"],
                    "target_answer": trial["answer"],
                    "conversation_id": trial["trajectory_id"],  # Use unique trajectory ID for reward isolation
                    "initial_memories": initial_memories,
                    "evidence": trial.get("evidence", []),
                    "category": trial.get("category", 0),
                    # Keep original sample_id for reference
                    "original_sample_id": trial["sample_id"],
                },
            }
            training_data.append(record)

        return training_data

    def _create_training_prompt(
        self, memories: List[Dict], target_question: str, conversation_context: Dict
    ) -> str:
        """Create training prompt for memory update agent."""

        # Format initial memories
        memory_text = "Initial Memory Database:\\n"
        for i, mem in enumerate(memories, 1):
            memory_text += f"{i}. [{mem['memory_type']}] {mem['content']} (from {mem['speaker']})\\n"

        # Create context about conversation
        context_text = f"\\nConversation Context:\\n"
        context_text += f"Number of sessions: {len([k for k in conversation_context.keys() if k.startswith('session')]) // 2}\\n"

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

        # Save as parquet - need to serialize complex nested structures
        # Even though verl expects native objects, parquet can't handle deeply nested dicts
        # So we serialize to JSON and will need to handle deserialization in verl
        import json

        def serialize_for_parquet(data):
            """Serialize only the problematic nested fields."""
            result = []
            for record in data:
                record_copy = record.copy()
                # Only serialize extra_info which has deep nesting
                if "extra_info" in record_copy:
                    record_copy["extra_info"] = json.dumps(record_copy["extra_info"])
                # Keep prompt and messages as native lists - verl can handle these
                result.append(record_copy)
            return result

        train_data_serialized = serialize_for_parquet(train_data)
        test_data_serialized = serialize_for_parquet(test_data)

        # Use pandas for saving
        train_df = pd.DataFrame(train_data_serialized)
        test_df = pd.DataFrame(test_data_serialized)

        train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
        test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

        print(
            f"✅ Saved {len(train_data)} training samples to {output_dir}/train.parquet"
        )
        print(f"✅ Saved {len(test_data)} test samples to {output_dir}/test.parquet")

    def process_full_pipeline(
        self, output_dir: str = "/data/users/alan/memupdate/data/locomo"
    ):
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
            "total_test_examples": len(test_df),
        }

        with open(output_path / "dataset_stats.json", "w") as f:
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
