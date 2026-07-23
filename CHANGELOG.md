# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-23

First release.

### Added

- `ParalinguisticAnalyzer` — extracts `ProsodyFeatures` (speaking rate, energy
  in dB, energy variance, pitch mean and variance, silence ratio) from a raw
  float32 mono audio array and returns an `EmotionResult` with an emotion
  label, confidence, top-3 candidates, arousal, valence, and an
  injection-ready `context_string`.
- Two backends: `"hubert"` (default, `superb/hubert-base-superb-er`, requires
  the `[full]` extra) and `"whisper"` — a pure-prosody heuristic classifier
  that needs nothing beyond numpy. Predictions below the 0.35 confidence
  threshold collapse to `"uncertain"`.
- `ConversationTracker` — keeps a rolling window (default 5 turns) of results
  and derives a `ConversationState`: mean valence and arousal, valence and
  arousal trends, turn-over-turn shift detection, and a
  `conversation_context_string` that includes behavioral guidance for the LLM.
- `EmologMiddleware` — the drop-in. `process()` runs analyzer plus tracker,
  `inject()` appends the context block to an OpenAI-style messages list without
  mutating the caller's list, `build_langchain_messages()` does the same for
  LangChain, and `get_tts_style_hint()` returns per-emotion stability, style,
  speaking rate, and a one-line TTS instruction as strings.
- `benchmark/eval_harness.py` — `BenchmarkSuite` scores agent replies across 6
  scenarios (angry, sad, fearful, happy, escalating, neutral) on empathy
  alignment, tone appropriateness, urgency calibration, and emotion-context
  utilization, and reports baseline vs. emolog-enabled.
- `examples/demo_before_after.py` — before/after system prompts and TTS hints
  for the angry, sad, and neutral scenarios.
- `playground/` — a self-contained single-page interactive demo driven by real
  SDK output.
- 25 tests covering prosody extraction, arousal/valence mapping, context-string
  formatting, the tracker, middleware injection, and the benchmark harness. CI
  runs them on Python 3.9 through 3.12.

### Notes

- No network calls anywhere in the core library, and no disk reads during
  `analyze()` — model loading happens only in `__init__`.
- Base install depends on numpy alone; `torch` and `transformers` are behind
  the `[full]` extra.

[0.1.0]: https://github.com/rishabhsabnavis/emolog/releases/tag/v0.1.0
