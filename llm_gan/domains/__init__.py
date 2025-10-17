"""Domain-specific configurations for training and evaluation."""

from .base import BaseDomain
from .creative_writing import CreativeWritingDomain
from .proofs import ProofsDomain
from .registry import DomainRegistry

__all__ = [
    'BaseDomain',
    'CreativeWritingDomain', 
    'ProofsDomain',
    'DomainRegistry'
]