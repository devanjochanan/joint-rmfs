"""RTS (Return-To-Storage) decision package.

Public surface:
    RTSDestinationContext  — input snapshot for the policy
    RTSDecision            — output from the policy
    RTSPolicy              — protocol / interface
    CurrentRTSPolicy       — behavior-preserving default implementation
"""

from .types import RTSDestinationContext, RTSDecision
from .policy import RTSPolicy
from .nearest_policy import CurrentRTSPolicy

__all__ = [
    "RTSDestinationContext",
    "RTSDecision",
    "RTSPolicy",
    "CurrentRTSPolicy",
]
