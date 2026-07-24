"""emolog — the missing emotional memory layer for voice agents."""

from .analyzer import (
    EMOTION_LABELS,
    EmotionResult,
    ParalinguisticAnalyzer,
    ProsodyFeatures,
)
from .middleware import EmologMiddleware
from .tracker import ConversationState, ConversationTracker

__version__ = "0.1.0"

__all__ = [
    "EMOTION_LABELS",
    "EmotionResult",
    "ParalinguisticAnalyzer",
    "ProsodyFeatures",
    "ConversationState",
    "ConversationTracker",
    "EmologMiddleware",
    "__version__",
]
