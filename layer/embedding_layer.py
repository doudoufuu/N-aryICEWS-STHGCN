import torch   
from torch import nn   
from typing import Tuple   


class CheckinEmbedding(nn.Module):  # [MODIFIED]
    """Project raw entity / event / chain features into a unified embedding space."""   

    def __init__(self, entity_dim: int, event_dim: int, chain_dim: int, embed_size: int, dropout: float) -> None:  # [MODIFIED]
        super().__init__()   
        self.entity_proj = nn.Sequential(   
            nn.Linear(entity_dim, embed_size),   
            nn.ReLU(),   
            nn.Dropout(dropout),   
        )   
        self.event_proj = nn.Sequential(   
            nn.Linear(event_dim, embed_size),   
            nn.ReLU(),   
            nn.Dropout(dropout),   
        )   
        self.chain_proj = nn.Sequential(   
            nn.Linear(chain_dim, embed_size),   
            nn.ReLU(),   
            nn.Dropout(dropout),   
        )   
        self.output_embed_size = embed_size   

    def forward(  # [MODIFIED]
        self,   
        entity_x: torch.Tensor,   
        event_x: torch.Tensor,   
        chain_x: torch.Tensor,   
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   
        entity_emb = self.entity_proj(entity_x)   
        event_emb = self.event_proj(event_x)   
        chain_emb = self.chain_proj(chain_x)   
        return entity_emb, event_emb, chain_emb   


class EdgeEmbedding(nn.Module):   
    """Edge type encoder reused from original design (kept minimal for compatibility)."""   

    def __init__(self, num_edge_type: int, embed_size: int) -> None:   
        super().__init__()   
        self.embedding = nn.Embedding(num_edge_type, embed_size)   
        self.output_embed_size = embed_size   

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:   
        return self.embedding(edge_type.long())   
