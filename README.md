# emolog

> **The missing emotional memory layer for voice agents.**

[![tests](https://github.com/rishabhsabnavis/emolog/actions/workflows/test.yml/badge.svg)](https://github.com/rishabhsabnavis/emolog/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/emolog.svg)](https://pypi.org/project/emolog/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<!-- 10. Top-of-README before/after demo GIF/video goes here once recorded. -->

## The problem

A voice agent pipeline converts audio to text before the LLM ever sees it, and
that conversion throws away everything except the words. When a user snaps
"I'm FINE" through gritted teeth, ASR hands the LLM `I'm fine` — the sarcasm,
the clipped delivery, the raised volume are all gone, and the agent cheerfully
replies "Great, glad to hear it!" emolog sits between the ASR and LLM steps,
reads the raw audio *before* that conversion happens, and tells the LLM what
the user actually sounded like.

## Before / after

Same ASR transcript in both cases — `"This is the third time I've called about
this charge."` Only the system prompt differs:

```text
--- BEFORE emolog ---
You are a helpful customer support assistant.

--- AFTER emolog ---
You are a helpful customer support assistant.

[Paralinguistic context: The user sounds angry (74% confidence). Speaking rate
is very fast. Vocal energy is very high. There is notable emphasis and stress
in the delivery. ]
```

The first prompt gets a policy recitation. The second gets an apology and an
escalation, because the LLM knows it is talking to someone who is already
angry. On multi-turn calls the tracker adds the arc, not just the moment:

```text
[Conversation emotional arc: The user currently sounds angry (76% confidence).
Overall conversation mood has been mixed. Overall energy level has been
moderate. The user's mood appears to be worsening. The user's energy appears to
be escalating. ALERT: Emotional state shifted from neutral to angry (worsened).
Respond with empathy and de-escalation. Acknowledge the user's frustration
before providing information. CAUTION: User appears increasingly distressed.
Prioritize resolution and avoid lengthy explanations. ]
```

Run it yourself: `python examples/demo_before_after.py`.

## Install

```bash
pip install emolog          # base: heuristic backend, numpy only
pip install emolog[full]    # adds torch/transformers for real HuBERT inference
```

The base install has one dependency (numpy) and runs the prosody-only heuristic
backend — no model download, no network. Use `[full]` when you want real model
inference from `superb/hubert-base-superb-er`, which is the default backend and
downloads weights on first use.

## Quickstart

```python
from emolog import EmologMiddleware
middleware = EmologMiddleware()
messages = middleware.inject(audio, sample_rate, transcription, messages)
```

`audio` is a float32 mono numpy array, `messages` is an OpenAI-style list.
`inject()` returns a copy with the paralinguistic context appended to the
system message; it never mutates the list you pass in. On a base install
(numpy only), pass `EmologMiddleware(backend="whisper")` to use the heuristic
backend instead of downloading HuBERT.

Two more things it gives you:

```python
emotion, state = middleware.process(audio, sample_rate, transcription)
emotion.emotion, emotion.confidence, state.valence_trend
# ('angry', 0.74, 'worsening')

middleware.get_tts_style_hint()
# {'stability': '0.25', 'style': 'calm and empathetic',
#  'speaking_rate': '0.85', 'description': "Speak in a calm, empathetic tone; ..."}
```

For LangChain, use `build_langchain_messages(...)` and pass your
`SystemMessage` / `HumanMessage` classes.

## Model-agnostic

emolog never calls an LLM or a TTS API. It reads audio in and returns a context
string and style hints out — that is the entire contract. It works with any LLM
(OpenAI, Anthropic, open-weight) and any TTS provider (ElevenLabs, Cartesia,
PlayHT, whatever you're on), because it only edits the prompt you were already
sending and hands you a dict you can map onto your own voice parameters. There
is no network call anywhere in the core library.

## How it works

```
[raw audio] ──► ParalinguisticAnalyzer ──► EmotionResult
                        │                       │
                        │                       ▼
                        │              ConversationTracker ──► ConversationState
                        │                       │
                        └───────────────────────┘
                                      │
                                      ▼
                              EmologMiddleware
                              (inject into LLM messages)
```

`ParalinguisticAnalyzer` extracts prosody (speaking rate, energy, pitch,
silence ratio) and an emotion label per turn. `ConversationTracker` keeps a
rolling window of those results and derives the arc — mean valence and arousal,
trend, and shift detection. `EmologMiddleware` is the drop-in that owns both and
writes the result into your prompt.

## Benchmark results

`benchmark/eval_harness.py` scores agent replies across 6 scenarios on empathy
alignment, tone appropriateness, urgency calibration, and emotion-context
utilization. Baseline vs. emolog-enabled:

| scenario | emotion | baseline | emolog |
| --- | --- | ---: | ---: |
| angry_001 | angry | 0.27 | **0.94** |
| sad_001 | sad | 0.40 | **1.00** |
| fearful_001 | fearful | 0.27 | **0.94** |
| happy_001 | happy | 0.53 | **0.90** |
| escalating_001 | frustrated | 0.33 | **0.94** |
| neutral_001 | neutral | 0.80 | **0.90** |
| **mean** | | **0.433** | **0.935** |
| **pass rate** | | 17% (1/6) | **100% (6/6)** |

That is a +116% lift on mean score. Read the number for what it is: the bundled
agent returns canned responses, so this run measures the harness and the
scoring rubric, not a specific production LLM. Pass your own model-backed
`agent_fn` to `BenchmarkSuite.run()` to measure yours.

## Limitations

- The heuristic backend is approximate. It infers emotion from prosody alone,
  and its confidence scores run low (0.35–0.50 is typical); below the 0.35
  threshold the label becomes `"uncertain"` rather than a guess.
- The HuBERT backend needs `pip install emolog[full]` plus a model download on
  first run, so the base install is the only zero-setup path.
- Prosody heuristics — speaking-rate bands, energy thresholds, the pitch proxy
  — are tuned on English. They may not transfer cleanly to tonal or other
  non-English speech yet.
- Pitch is estimated with a zero-crossing-rate proxy, not a real F0 tracker, to
  keep numpy as the only base dependency. It tracks relative change well and
  absolute hertz poorly.
- Emotion recognition from voice is noisy in general. Treat the injected
  context as a signal to the LLM, not a verdict about the user.

## License

MIT — see [LICENSE](LICENSE).
