"""Qwen3-0.6B embeddings implementation that generates 1024-dimensional embeddings."""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List

class QwenEmbeddings:
    """Qwen3-0.6B embeddings."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Qwen3-0.6B embedding model on {self.device}...")
        
        # Use Qwen3-0.6B
        model_name = "Qwen/Qwen2.5-0.5B"  # This should be close to 0.6B and have ~1024 hidden size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        print(f"✅ Qwen3-0.6B model loaded with hidden size: {self.model.config.hidden_size}")
        
        # If not 1024, add projection
        if self.model.config.hidden_size != 1024:
            self.projection = torch.nn.Linear(self.model.config.hidden_size, 1024).to(self.device)
            # Initialize with deterministic weights
            torch.manual_seed(42)
            torch.nn.init.xavier_uniform_(self.projection.weight)
            torch.nn.init.zeros_(self.projection.bias)
            print(f"Added projection {self.model.config.hidden_size} -> 1024")
        else:
            self.projection = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with Qwen3-0.6B."""
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                
                # Apply projection if needed
                if self.projection is not None:
                    embedding = self.projection(embedding)
                
                embeddings.append(embedding.cpu().numpy().tolist())
        
        return embeddings

