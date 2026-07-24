"""Demo: LLM system prompts with and without emolog's injected context.

Runs three scenarios — angry, sad, neutral — through the middleware and prints,
for each, the system prompt the LLM would see BEFORE emolog and AFTER, plus the
TTS style hint emolog derives. Uses the heuristic ("whisper") backend so it runs
on the base install with no model download and no network.

The audio here is synthesized, not recorded: each `_synth_*` helper shapes a
tone whose prosody (loudness, pace, pitch, emphasis) lands in the region the
heuristic reads as that emotion. It stands in for a real ASR front-end feeding
raw audio to emolog — the point is the injected context, not the audio realism.
"""

from __future__ import annotations

import numpy as np

from emologcontext import EmologMiddleware

SR = 16000
BASE_SYSTEM_PROMPT = "You are a helpful customer support assistant."


# --------------------------------------------------------------------------- #
# Audio synthesis
# --------------------------------------------------------------------------- #
def _synth(
    freq: float,
    dur: float,
    amp: float,
    n_syllables: int,
    floor: float,
    sr: int = SR,
) -> np.ndarray:
    """A carrier tone at `freq` Hz, amplitude-modulated by `n_syllables` bumps.

    - `freq` sets pitch (the ZCR proxy tracks carrier frequency).
    - `amp` sets loudness (energy_db).
    - `n_syllables` over `dur` sets the syllable-peak rate.
    - `floor` is the between-syllable amplitude: a low floor means sharp,
      emphatic bumps (high energy variance); a high floor means a flat, even
      delivery (low variance).
    """
    t = np.arange(int(dur * sr)) / sr
    carrier = np.sin(2 * np.pi * freq * t)

    centers = np.linspace(0.0, dur, n_syllables, endpoint=False) + dur / (2 * n_syllables)
    width = dur / (n_syllables * 3.0)
    env = np.zeros_like(t)
    for c in centers:
        env += np.exp(-0.5 * ((t - c) / width) ** 2)
    env = env / env.max()
    env = floor + (1.0 - floor) * env

    return (amp * carrier * env).astype(np.float32)


def _synth_angry(sr: int = SR) -> np.ndarray:
    """Loud, fast, emphatic: high energy, high rate, sharp syllable bursts."""
    return _synth(freq=175.0, dur=2.2, amp=0.9, n_syllables=9, floor=0.05, sr=sr)


def _synth_sad(sr: int = SR) -> np.ndarray:
    """Quiet, slow, low-pitched, with trailing pauses."""
    audio = _synth(freq=80.0, dur=4.2, amp=0.02, n_syllables=3, floor=0.2, sr=sr)
    # A stretch of near-silence at the end -> hesitation / high silence ratio.
    return np.concatenate([audio, np.zeros(int(1.3 * sr), dtype=np.float32)])


def _synth_neutral(sr: int = SR) -> np.ndarray:
    """Moderate loudness, steady pace, even delivery (little emphasis)."""
    return _synth(freq=140.0, dur=2.0, amp=0.08, n_syllables=6, floor=0.55, sr=sr)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
SCENARIOS = [
    {
        "label": "ANGRY — repeat caller, unresolved billing",
        "audio": _synth_angry,
        "transcript": "This is the third time I've called about this charge.",
    },
    {
        "label": "SAD — distressed user on a support call",
        "audio": _synth_sad,
        "transcript": "I don't really know where to start.",
    },
    {
        "label": "NEUTRAL — simple informational query",
        "audio": _synth_neutral,
        "transcript": "What are your support hours?",
    },
]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _rule(char: str = "=") -> str:
    return char * 74


def _run_scenario(middleware: EmologMiddleware, scenario: dict) -> None:
    audio = scenario["audio"]()
    transcript = scenario["transcript"]

    # Reset so each scenario shows its own single-turn context, not an arc
    # accumulated across unrelated scenarios.
    middleware.reset()

    before = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    after = middleware.inject(audio, SR, transcript, before)

    result = middleware.last_emotion_result
    hint = middleware.get_tts_style_hint()

    print(_rule())
    print(scenario["label"])
    print(_rule())
    print(f'ASR transcript (all the LLM would normally get):\n  "{transcript}"')
    print(
        f"\nWhat emolog heard: {result.emotion} "
        f"({result.confidence * 100:.0f}% confidence), "
        f"arousal={result.arousal}, valence={result.valence}"
    )

    print("\n--- BEFORE emolog: system prompt the LLM sees ---")
    print(f"  {before[0]['content']}")

    print("\n--- AFTER emolog: system prompt the LLM sees ---")
    for line in after[0]["content"].splitlines():
        print(f"  {line}")

    print("\n--- TTS style hint (for the reply's voice) ---")
    print(
        f"  style={hint['style']!r}  rate={hint['speaking_rate']}  "
        f"stability={hint['stability']}"
    )
    print(f"  {hint['description']}")
    print()


def main() -> None:
    # Heuristic backend: no model download, no network, numpy only.
    middleware = EmologMiddleware(backend="whisper")

    print("\nemolog before/after demo")
    print(
        "Same ASR text in every case. Only emolog sees the audio underneath it,\n"
        "and only the AFTER prompt tells the LLM how the user actually sounded.\n"
    )
    for scenario in SCENARIOS:
        _run_scenario(middleware, scenario)


if __name__ == "__main__":
    main()
