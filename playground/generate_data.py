"""Generate the playground's data by running the REAL emolog SDK.

The design handoff shipped hardcoded, illustrative per-line numbers. This script
replaces them with genuine output: it synthesizes audio for each sample line,
runs it through ParalinguisticAnalyzer -> ConversationTracker -> EmologMiddleware
(the heuristic backend, no model download, no network), and runs the actual
benchmark harness. It writes `data.js`, which `index.html` loads.

Regenerate any time the SDK changes:

    python playground/generate_data.py

What is real vs. illustrative:
  - emotion / confidence / arousal / valence  -> real analyzer output
  - contextString (injected prompt block)      -> real _build_context_block()
  - convoMood / convoDesc                      -> derived from real ConversationState
  - tts.style / tts.rate / tts.stability       -> real get_tts_style_hint()
  - benchmark bars                             -> real BenchmarkSuite run
  - transcript / baselineReply / awareReply    -> illustrative copy (the SDK
                                                  reads audio; it does not write
                                                  transcripts or LLM replies)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make `emolog` and `benchmark` importable when run from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from emologcontext import EmologMiddleware  # noqa: E402
from benchmark.eval_harness import (  # noqa: E402
    SCENARIOS,
    BenchmarkSuite,
    _bucket,
    _score_context_use,
    _score_empathy,
    _score_tone,
    _score_urgency,
    support_agent,
)

SR = 16000


# --------------------------------------------------------------------------- #
# Audio synthesis (same technique as examples/demo_before_after.py: a carrier
# tone amplitude-modulated by syllable bumps, tuned so the heuristic classifier
# lands each clip in the intended emotion region).
# --------------------------------------------------------------------------- #
def _synth(freq, dur, amp, n_syllables, floor, sr=SR):
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


def _angry():
    return _synth(175.0, 2.2, 0.9, 9, 0.05)


def _clipped():
    # Short, punchy, mid-high energy: a terse "I'm fine." with the irritation
    # in the delivery. Reads angry with a coherent (not "very slow") rate.
    return _synth(165.0, 0.7, 0.5, 2, 0.1)


def _sad():
    body = _synth(80.0, 4.2, 0.02, 3, 0.2)
    return np.concatenate([body, np.zeros(int(1.3 * SR), dtype=np.float32)])


def _neutral():
    return _synth(140.0, 2.0, 0.08, 6, 0.55)


def _happy():
    return _synth(240.0, 2.0, 0.5, 10, 0.25)


# --------------------------------------------------------------------------- #
# Derive the Tracker card's short label/description from the real state.
# --------------------------------------------------------------------------- #
def _convo_labels(state):
    if state.turn_count <= 1:
        return "Single turn", "Not enough history yet for a conversational arc."

    if state.arousal_trend == "escalating":
        mood = f"Escalating — {state.turn_count} turns"
    elif state.valence_trend == "improving":
        mood = f"Improving — {state.turn_count} turns"
    elif state.valence_trend == "worsening":
        mood = f"Worsening — {state.turn_count} turns"
    else:
        mood = f"Stable — {state.turn_count} turns"

    if state.shift_detected:
        desc = state.shift_description
    else:
        energy = (
            "high-energy" if state.mean_arousal > 0.65
            else "moderate" if state.mean_arousal > 0.35 else "low-energy"
        )
        valence = (
            "positive" if state.mean_valence > 0.65
            else "mixed" if state.mean_valence > 0.35 else "negative"
        )
        desc = f"Mood {valence}, {energy}. No sharp shift yet."
    return mood, desc


# --------------------------------------------------------------------------- #
# Preset specs: intended narrative + how to drive the SDK to produce it.
# `turns` is the audio sequence fed to one middleware instance; the LAST turn is
# the one whose transcript/analysis the card displays. Multi-turn presets build
# a real ConversationTracker arc.
# --------------------------------------------------------------------------- #
_PRESET_SPECS = [
    {
        "id": "sarcastic",
        "tag": 'flat "I\'m fine."',
        "transcript": "I'm fine.",
        "turns": [(_clipped, "I'm fine.")],
        "baselineReply": "Great, glad to hear it! Is there anything else I can help you with today?",
        "awareReply": "I want to make sure this is actually resolved — you sound a little frustrated. Let's get it fixed properly.",
    },
    {
        "id": "billing",
        "tag": "angry, escalating",
        "transcript": "This is the THIRD time I've called about this.",
        # neutral x3 then angry -> real worsening valence, escalating arousal,
        # a neutral->angry shift on the final turn, and a CAUTION flag.
        "turns": [
            (_neutral, "Hi, I need to sort out a billing error."),
            (_neutral, "No, that's still not right."),
            (_neutral, "I already gave you that twice."),
            (_angry, "This is the THIRD time I've called about this."),
        ],
        "baselineReply": "I understand, let me pull up your account. Can I get your account number please?",
        "awareReply": "I hear you — three calls for the same issue is genuinely frustrating. I'm escalating this right now so it gets fixed immediately.",
    },
    {
        "id": "sad",
        "tag": "quiet, drained",
        "transcript": "I don't really know what to do anymore.",
        "turns": [
            (_sad, "It's been a really hard week."),
            (_sad, "I don't really know what to do anymore."),
        ],
        "baselineReply": "Okay. Here's the information you requested. Let me know if you need anything else.",
        "awareReply": "Take your time — I'm here with you. Let's work through this together, one step at a time.",
    },
    {
        "id": "happy",
        "tag": "delighted",
        "transcript": "Oh my gosh, it arrived already?! That's amazing!",
        "turns": [(_happy, "Oh my gosh, it arrived already?! That's amazing!")],
        "baselineReply": "Your order has been marked as delivered. Please rate your experience below.",
        "awareReply": "That's so great to hear! Hope it's exactly what you wanted — let us know if anything needs a follow-up.",
    },
    {
        "id": "neutral",
        "tag": "just a question",
        "transcript": "Can you check my account balance?",
        "turns": [(_neutral, "Can you check my account balance?")],
        "baselineReply": "Sure, one moment while I pull that up.",
        "awareReply": "Sure, one moment while I pull that up.",
    },
]


def _build_presets(middleware: EmologMiddleware) -> list[dict]:
    presets = []
    for spec in _PRESET_SPECS:
        middleware.reset()
        emotion_result = None
        conv_state = None
        for audio_fn, transcript in spec["turns"]:
            emotion_result, conv_state = middleware.process(audio_fn(), SR, transcript)

        context_block = middleware._build_context_block(emotion_result, conv_state)
        hint = middleware.get_tts_style_hint()
        convo_mood, convo_desc = _convo_labels(conv_state)

        presets.append(
            {
                "id": spec["id"],
                "tag": spec["tag"],
                "transcript": spec["transcript"],
                "emotion": emotion_result.emotion,
                "confidence": round(emotion_result.confidence * 100),
                "arousal": emotion_result.arousal,
                "valence": emotion_result.valence,
                "contextString": context_block,
                "convoMood": convo_mood,
                "convoDesc": convo_desc,
                "tts": {
                    "style": hint["style"],
                    "rate": hint["speaking_rate"],
                    "stability": hint["stability"],
                },
                "baselineReply": spec["baselineReply"],
                "awareReply": spec["awareReply"],
            }
        )
    return presets


# --------------------------------------------------------------------------- #
# Benchmark: real per-dimension means, emolog vs baseline, over the 6 scenarios.
# For a fair paired bar we score BOTH runs on all four dimensions (the harness
# normally omits emotion-context-utilization from the baseline overall; here we
# still measure how much the baseline replies happen to use it).
# --------------------------------------------------------------------------- #
_DIM_LABELS = [
    ("empathy_alignment", "Empathy alignment"),
    ("tone_appropriateness", "Tone appropriateness"),
    ("urgency_calibration", "Urgency calibration"),
    ("emotion_context_utilization", "Emotion-context use"),
]


def _score_all_dims(scenario: dict, response: str, emolog_enabled: bool) -> dict:
    emotion = scenario["emotion"]
    bucket = _bucket(emotion)
    wc = len(response.split())
    return {
        "empathy_alignment": _score_empathy(emotion, response),
        "tone_appropriateness": _score_tone(bucket, response),
        "urgency_calibration": _score_urgency(bucket, wc, emolog_enabled),
        "emotion_context_utilization": _score_context_use(bucket, response),
    }


def _build_benchmark() -> list[dict]:
    aware_totals = {k: 0.0 for k, _ in _DIM_LABELS}
    base_totals = {k: 0.0 for k, _ in _DIM_LABELS}
    n = len(SCENARIOS)

    for scenario in SCENARIOS:
        aware_resp = support_agent(scenario["transcript"], scenario["emotion"], scenario["context"])
        base_resp = support_agent(scenario["transcript"], None, scenario["context"])
        aware = _score_all_dims(scenario, aware_resp, True)
        base = _score_all_dims(scenario, base_resp, False)
        for k, _ in _DIM_LABELS:
            aware_totals[k] += aware[k]
            base_totals[k] += base[k]

    return [
        {
            "name": label,
            "baseline": round(base_totals[key] / n, 3),
            "emolog": round(aware_totals[key] / n, 3),
        }
        for key, label in _DIM_LABELS
    ]


def main() -> None:
    middleware = EmologMiddleware(backend="whisper")  # heuristic, numpy-only
    presets = _build_presets(middleware)
    benchmark = _build_benchmark()

    suite = BenchmarkSuite()
    lift = (
        suite.run(support_agent, emolog_enabled=True).mean_score
        - suite.run(support_agent, emolog_enabled=False).mean_score
    )

    payload = {"presets": presets, "benchmark": benchmark, "meanLift": round(lift, 3)}
    out = Path(__file__).resolve().parent / "data.js"
    out.write_text(
        "// AUTO-GENERATED by generate_data.py from the real emolog SDK. Do not edit by hand.\n"
        "window.EMOLOG_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )

    print(f"Wrote {out}")
    for p in presets:
        print(f"  {p['id']:10} -> {p['emotion']:9} {p['confidence']:>3}%  "
              f"arousal={p['arousal']:<6} | {p['convoMood']}")
    print("  benchmark:")
    for d in benchmark:
        print(f"    {d['name']:22} baseline {d['baseline']:.2f}  emolog {d['emolog']:.2f}")


if __name__ == "__main__":
    main()
