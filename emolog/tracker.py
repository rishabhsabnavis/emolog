"""Conversation-level emotional arc tracking.

Maintains the emotional arc of a multi-turn conversation. Each call to
``update()`` takes an ``EmotionResult`` and returns a ``ConversationState``.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Literal

from .analyzer import _EMOTION_AXES, EmotionResult

# Derived from the analyzer's _EMOTION_AXES table rather than restated here.
#
# These were previously hand-written nine-label dicts, and they had drifted from
# the taxonomy the analyzer can actually emit. Every label present in
# _EMOTION_AXES but absent from the dicts (`frustrated`, `distressed`,
# `anxious`, `contempt`, `disappointed`, …) fell through to the 0.5/0.4 default
# — numerically identical to `neutral`. A caller who stayed *frustrated* for
# three turns therefore reported "Overall conversation mood has been mixed" with
# no de-escalation guidance at all, which is precisely the case emolog exists to
# catch. `disgusted` was also missing from the arousal dict outright, despite
# being a label the shipped heuristic backend emits.
#
# Deriving both tables keeps the tracker in step with the taxonomy as it grows,
# which is the stated design intent in analyzer.py's module docstring.
_VALENCE_SCORE = {label: axes[0] for label, axes in _EMOTION_AXES.items()}
_AROUSAL_SCORE = {label: axes[1] for label, axes in _EMOTION_AXES.items()}

# Fallback for a label outside the taxonomy entirely — e.g. a HuBERT checkpoint
# whose id2label set does not match ours. Same coordinates as `neutral`.
_DEFAULT_VALENCE = 0.5
_DEFAULT_AROUSAL = 0.4


@dataclass
class ConversationState:
    current_emotion: str
    current_confidence: float
    mean_valence: float
    mean_arousal: float
    valence_trend: Literal["improving", "stable", "worsening"]
    arousal_trend: Literal["escalating", "stable", "de-escalating"]
    shift_detected: bool
    shift_description: str
    turn_count: int
    # Computed in __post_init__, NOT a constructor arg.
    conversation_context_string: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.conversation_context_string = _build_conversation_context(self)


class ConversationTracker:
    def __init__(self, window: int = 5, shift_threshold: float = 0.3) -> None:
        self.window = window
        self.shift_threshold = shift_threshold
        self._history: collections.deque[EmotionResult] = collections.deque(
            maxlen=window
        )

    def update(self, result: EmotionResult) -> ConversationState:
        """Append result to history, then return self._compute_state()."""
        self._history.append(result)
        return self._compute_state()

    def reset(self) -> None:
        """Clear self._history."""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._history)

    def _compute_state(self) -> ConversationState:
        history = list(self._history)
        valences = [_VALENCE_SCORE.get(r.emotion, _DEFAULT_VALENCE) for r in history]
        arousals = [_AROUSAL_SCORE.get(r.emotion, _DEFAULT_AROUSAL) for r in history]

        n = len(history)
        mean_valence = sum(valences) / n
        mean_arousal = sum(arousals) / n

        current = history[-1]

        # Trend: compare first half vs second half of the history window.
        valence_trend: Literal["improving", "stable", "worsening"] = "stable"
        arousal_trend: Literal["escalating", "stable", "de-escalating"] = "stable"
        if n >= 3:
            half = n // 2
            early_v = sum(valences[:half]) / half
            late_v = sum(valences[-half:]) / half
            dv = late_v - early_v
            if dv > 0.15:
                valence_trend = "improving"
            elif dv < -0.15:
                valence_trend = "worsening"

            early_a = sum(arousals[:half]) / half
            late_a = sum(arousals[-half:]) / half
            da = late_a - early_a
            if da > 0.15:
                arousal_trend = "escalating"
            elif da < -0.15:
                arousal_trend = "de-escalating"

        # Shift detection: compare the last two turns' valence scores.
        shift_detected = False
        shift_description = ""
        if n >= 2:
            prev, curr = history[-2], history[-1]
            prev_v = _VALENCE_SCORE.get(prev.emotion, _DEFAULT_VALENCE)
            curr_v = _VALENCE_SCORE.get(curr.emotion, _DEFAULT_VALENCE)
            if abs(curr_v - prev_v) >= self.shift_threshold:
                shift_detected = True
                direction = "worsened" if curr_v < prev_v else "improved"
                shift_description = (
                    f"Emotional state shifted from {prev.emotion} to "
                    f"{curr.emotion} ({direction})."
                )

        return ConversationState(
            current_emotion=current.emotion,
            current_confidence=current.confidence,
            mean_valence=mean_valence,
            mean_arousal=mean_arousal,
            valence_trend=valence_trend,
            arousal_trend=arousal_trend,
            shift_detected=shift_detected,
            shift_description=shift_description,
            turn_count=n,
        )


def _build_conversation_context(state: ConversationState) -> str:
    """Wrapped in '[Conversation emotional arc: ... ]'. Include current emotion,
    mood/energy descriptions, non-stable trend lines, shift alert, and the
    per-emotion LLM behavioral guidance block."""
    parts: list[str] = []

    parts.append(
        f"The user currently sounds {state.current_emotion} "
        f"({round(state.current_confidence * 100)}% confidence)."
    )

    if state.mean_valence > 0.65:
        mood = "positive"
    elif state.mean_valence > 0.35:
        mood = "mixed"
    else:
        mood = "negative"
    parts.append(f"Overall conversation mood has been {mood}.")

    if state.mean_arousal > 0.65:
        energy = "high-energy"
    elif state.mean_arousal > 0.35:
        energy = "moderate"
    else:
        energy = "low-energy"
    parts.append(f"Overall energy level has been {energy}.")

    if state.valence_trend != "stable":
        parts.append(f"The user's mood appears to be {state.valence_trend}.")
    if state.arousal_trend != "stable":
        parts.append(f"The user's energy appears to be {state.arousal_trend}.")

    if state.shift_detected:
        parts.append(f"ALERT: {state.shift_description}")

    # Behavioral guidance keys off the emotion's (valence, arousal) coordinates
    # rather than a hardcoded label set, so every label in the taxonomy gets a
    # guidance line instead of only the nine that were spelled out here. The
    # thresholds reproduce the old label sets exactly for those nine: negative
    # valence (< 0.4 — the analyzer's own "negative" cutoff in
    # _estimate_valence) splits at arousal 0.5 into activated distress
    # (angry / fearful / disgusted, and now frustrated / anxious / distressed /
    # contempt) and deactivated distress (sad, and now disappointed / shame /
    # guilt). The positive branch requires real activation (arousal >= 0.6),
    # which is why `calm` and `neutral` still get no guidance line.
    valence = _VALENCE_SCORE.get(state.current_emotion, _DEFAULT_VALENCE)
    arousal = _AROUSAL_SCORE.get(state.current_emotion, _DEFAULT_AROUSAL)

    if valence < 0.4 and arousal >= 0.5:
        parts.append(
            "Respond with empathy and de-escalation. Acknowledge the user's "
            "frustration before providing information."
        )
    elif valence < 0.4:
        parts.append(
            "Respond with warmth and patience. Avoid overly upbeat or "
            "transactional language."
        )
    elif (
        valence >= 0.5
        and arousal >= 0.6
        and state.valence_trend != "worsening"
    ):
        parts.append(
            "User is in a positive state. Match their energy with a warm, "
            "engaged tone."
        )

    if state.arousal_trend == "escalating" and state.valence_trend == "worsening":
        parts.append(
            "CAUTION: User appears increasingly distressed. Prioritize "
            "resolution and avoid lengthy explanations."
        )

    return "[Conversation emotional arc: " + " ".join(parts) + " ]"
