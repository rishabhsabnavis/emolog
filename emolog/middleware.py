"""The drop-in middleware: analyzer + tracker + injection helpers.

Constraints: inject() must NOT mutate the caller's messages list. All values
returned by get_tts_style_hint() must be strings.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .analyzer import _EMOTION_AXES, EmotionResult, ParalinguisticAnalyzer
from .tracker import ConversationState, ConversationTracker

# emotion -> (stability, style, speaking_rate, description)
_TTS_STYLE = {
    "angry": (0.25, "calm and empathetic", 0.85,
              "Speak in a calm, empathetic tone; slow down and acknowledge the user's frustration."),
    "sad": (0.30, "warm and gentle", 0.85,
            "Speak warmly and gently, with unhurried patience."),
    "fearful": (0.30, "reassuring", 0.90,
                "Speak in a steady, reassuring tone that conveys safety."),
    "happy": (0.40, "warm and engaged", 1.05,
              "Speak with warm, upbeat energy that matches the user's mood."),
    "surprised": (0.40, "attentive", 1.00,
                  "Speak attentively, ready to clarify or elaborate."),
    "disgusted": (0.25, "professional", 0.90,
                  "Speak in a composed, professional tone."),
    "calm": (0.50, "measured", 1.00,
             "Speak in a measured, even tone."),
    "neutral": (0.50, "natural", 1.00,
                "Speak naturally."),
    "uncertain": (0.50, "natural", 1.00,
                  "Speak naturally."),
}

# Taxonomy family -> which _TTS_STYLE entry to borrow when a label has none of
# its own. Without this, every label outside the nine above fell through to
# "neutral" — so a caller detected as `frustrated` got rate 1.0 and "Speak
# naturally.", i.e. the agent answered an angry-adjacent user in its default
# voice. Exact label wins; family is the fallback. This mirrors `_bucket()` in
# benchmark/eval_harness.py, which already resolves labels the same way.
_STYLE_BY_FAMILY = {
    "negative-high": "angry",
    "negative-low": "sad",
    "positive-high": "happy",
    "positive-low": "calm",
    "neutral": "neutral",
    "surprise": "surprised",
    "engaged": "surprised",
    "disengaged": "calm",
}


def _style_for(emotion: str) -> tuple[float, str, float, str]:
    """Resolve an emotion to a TTS style: exact label, then taxonomy family,
    then neutral for a label from outside the taxonomy altogether."""
    if emotion in _TTS_STYLE:
        return _TTS_STYLE[emotion]
    axes = _EMOTION_AXES.get(emotion)
    family = axes[2] if axes is not None else "neutral"
    return _TTS_STYLE[_STYLE_BY_FAMILY.get(family, "neutral")]


class EmologMiddleware:
    def __init__(
        self,
        backend: str = "hubert",
        model_name: str = "superb/hubert-base-superb-er",
        include_turn_context: bool = True,
        include_conv_context: bool = True,
        system_prompt_prefix: str = "",
        window: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.include_turn_context = include_turn_context
        self.include_conv_context = include_conv_context
        self.system_prompt_prefix = system_prompt_prefix

        self._analyzer = ParalinguisticAnalyzer(
            backend=backend, model_name=model_name, device=device
        )
        self._tracker = ConversationTracker(window=window)
        self.last_emotion_result: Optional[EmotionResult] = None
        self.last_conversation_state: Optional[ConversationState] = None

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        transcription: Optional[str] = None,
    ) -> tuple[EmotionResult, ConversationState]:
        """analyze → tracker.update → store both on self → return both."""
        emotion_result = self._analyzer.analyze(audio, sample_rate, transcription)
        conv_state = self._tracker.update(emotion_result)
        self.last_emotion_result = emotion_result
        self.last_conversation_state = conv_state
        return emotion_result, conv_state

    def inject(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcription: Optional[str],
        messages: list[dict],
    ) -> list[dict]:
        """OpenAI-style injection. Copy messages (do NOT mutate caller's list),
        append context block to the system message or insert one at index 0."""
        emotion_result, conv_state = self.process(audio, sample_rate, transcription)
        context_block = self._build_context_block(emotion_result, conv_state)

        # Copy so the caller's list is never mutated.
        messages = [dict(m) for m in messages]

        for message in messages:
            if message.get("role") == "system":
                existing = message.get("content", "")
                sep = "\n\n" if existing else ""
                message["content"] = existing + sep + context_block
                break
        else:
            messages.insert(0, {"role": "system", "content": context_block})

        return messages

    def build_langchain_messages(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcription: Optional[str],
        history: list,
        system_message_class=None,
        human_message_class=None,
    ) -> list:
        """[system_msg] + history + [human_msg(transcription)]. Use provided
        message classes if given, else return raw dicts."""
        emotion_result, conv_state = self.process(audio, sample_rate, transcription)
        context_block = self._build_context_block(emotion_result, conv_state)

        full_system = (self.system_prompt_prefix + "\n\n" + context_block).strip()
        human_content = transcription or ""

        if system_message_class is not None and human_message_class is not None:
            system_msg = system_message_class(content=full_system)
            human_msg = human_message_class(content=human_content)
        else:
            system_msg = {"role": "system", "content": full_system}
            human_msg = {"role": "user", "content": human_content}

        return [system_msg] + list(history) + [human_msg]

    def reset(self) -> None:
        """tracker.reset(); clear last_emotion_result / last_conversation_state."""
        self._tracker.reset()
        self.last_emotion_result = None
        self.last_conversation_state = None

    def get_tts_style_hint(self) -> dict[str, str]:
        """Map current emotion → {stability, style, speaking_rate, description},
        all as strings. Defaults when no state. Apply escalation override when
        arousal escalating AND valence worsening."""
        if self.last_conversation_state is None:
            return {
                "stability": "0.5",
                "style": "neutral",
                "speaking_rate": "1.0",
                "description": "Speak naturally.",
            }

        state = self.last_conversation_state
        stability, style, rate, description = _style_for(state.current_emotion)

        if (
            state.arousal_trend == "escalating"
            and state.valence_trend == "worsening"
        ):
            rate = max(0.80, rate - 0.10)
            stability = 0.20

        return {
            "stability": f"{stability}",
            "style": style,
            "speaking_rate": f"{rate}",
            "description": description,
        }

    def _build_context_block(
        self, emotion_result: EmotionResult, conv_state: ConversationState
    ) -> str:
        """Join (with double newline) the turn context string and, when
        conv_state.turn_count > 1, the conversation context string."""
        blocks: list[str] = []
        if self.include_turn_context:
            blocks.append(emotion_result.context_string)
        if self.include_conv_context and conv_state.turn_count > 1:
            blocks.append(conv_state.conversation_context_string)
        return "\n\n".join(blocks)
