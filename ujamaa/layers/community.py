"""
Ujamaa Community Routing — Core Innovation

Tokens cooperate, share resources, and lift each other up.
Built by Brian Tushae Thomas for Anthos Intelligence Company.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class TokenCollective(nn.Module):
    """
    Tokens vote on how computation should be allocated.
    Hard tokens (high uncertainty) get help from easy tokens.
    """

    def __init__(self, dim: int, window: int = 512):
        super().__init__()
        self.dim = dim
        self.window = window

        self.need_projection = nn.Linear(dim, 1)
        self.community_projection = nn.Linear(dim, dim)
        self.vote_aggregator = nn.Linear(window, window)

        self.shared_resource_pool = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.resource_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

        self.last_votes: Optional[torch.Tensor] = None
        self.last_allocation: Optional[torch.Tensor] = None

    def forward(self, token_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_states: [batch, seq, dim]
        Returns:
            adjusted_states: [batch, seq, dim]
        """
        batch, seq, dim = token_states.shape

        need_scores = torch.sigmoid(self.need_projection(token_states))  # [batch, seq, 1]

        # Pad or trim to window size for aggregation
        if seq <= self.window:
            padded = F.pad(need_scores.squeeze(-1), (0, self.window - seq))
            community_need = self.vote_aggregator(padded)[:, :seq].unsqueeze(-1)
        else:
            community_need = self.vote_aggregator(need_scores[:, :self.window].squeeze(-1))
            community_need = F.pad(community_need, (0, seq - self.window)).unsqueeze(-1)

        community_need = torch.sigmoid(community_need)

        resource_allocation = self.resource_gate(community_need - need_scores)
        shared_boost = resource_allocation * self.shared_resource_pool
        adjusted_states = token_states + shared_boost

        self.last_votes = need_scores.detach()
        self.last_allocation = resource_allocation.detach()

        return adjusted_states


class CommunityGate(nn.Module):
    """
    Community gate applied at each layer and globally.
    Provides coordination across all tokens.
    """

    def __init__(self, dim: int, window: int = 512):
        super().__init__()
        self.collective = TokenCollective(dim, window)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            adjusted_x: [batch, seq, dim]
            stats: routing information dict
        """
        adjusted = self.collective(x)
        adjusted = self.norm(adjusted + x)

        stats = {
            "community_votes": self.collective.last_votes,
            "community_allocation": self.collective.last_allocation,
            "community_impact": (adjusted - x).abs().mean().item(),
        }

        return adjusted, stats
