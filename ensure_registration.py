#!/usr/bin/env python3
"""
Ensure MemoryRewardManager is registered in Ray workers.
This script is imported by Ray workers to register the reward manager.
"""
import sys
import os

# Add memupdate to path
sys.path.insert(0, '/workspace/memupdate')

# Import memupdate to trigger registration
try:
    import memupdate
    print(f"✅ MemoryRewardManager registered in worker process {os.getpid()}")
except Exception as e:
    print(f"❌ Failed to register MemoryRewardManager in worker {os.getpid()}: {e}")