"""
UJAMAA MULTI-MODAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Built by Brian Tushae Thomas for Anthos Intelligence Company
© 2024-2025 Anthos Intelligence. All rights reserved.

Ujamaa: A cooperative, community-driven multi-modal foundation model.
Tokens cooperate, share resources, and lift each other up.

Architecture:
- Gated Recurrent Attention with community routing
- Multi-modal encoders (Vision, Audio)
- Mixture of Experts with dynamic routing
- Scalable growth from 1.5B to 100B+ parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from .model import UjamaaMultiModal, ujamaa_mm
from .config import UjamaaConfig
from .layers.community import CommunityGate, TokenCollective
from .layers.moe import MixtureOfExperts, Expert

__all__ = [
    "UjamaaMultiModal",
    "UjamaaConfig",
    "CommunityGate",
    "TokenCollective",
    "MixtureOfExperts",
    "Expert",
    "ujamaa_mm",
]

__version__ = "0.1.0"
__author__ = "Brian Tushae Thomas"
__company__ = "Anthos Intelligence"
__copyright__ = "© 2024-2025 Anthos Intelligence Company"
