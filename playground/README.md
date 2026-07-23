# emolog playground

A single-page interactive demo of emolog: pick one of five sample voice lines
and watch the pipeline (`ParalinguisticAnalyzer → ConversationTracker →
EmologMiddleware`) animate — showing the detected emotion, the injected system
prompt, the TTS style hint, and a baseline-vs-emolog reply comparison, then a
benchmark section.

Built from the `design_handoff_emolog_playground` Cream design, recreated as a
self-contained page (no design-tool runtime, no framework, no build step).

## Run it

```bash
# 1. (Re)generate the data from the real SDK — needs the base install (numpy).
python playground/generate_data.py

# 2. Serve the folder (needed so the browser can load data.js and the logo).
python -m http.server 8000 --directory playground
# then open http://localhost:8000
```

Opening `index.html` directly via `file://` also works in most browsers, but a
local server is the reliable path (some browsers block `data.js` over `file://`).

## What's real vs. illustrative

Unlike the original design mockup, which shipped hardcoded numbers, this page is
driven by `data.js`, which `generate_data.py` produces by **running the actual
emolog SDK**:

| Shown on the page | Source |
| --- | --- |
| emotion, confidence, arousal, valence | real `ParalinguisticAnalyzer` output |
| injected system-prompt block | real `EmologMiddleware._build_context_block()` |
| conversation mood / description | derived from the real `ConversationState` |
| TTS style / rate / stability | real `get_tts_style_hint()` |
| benchmark bars | real `BenchmarkSuite` run over the 6 scenarios |
| transcript, baseline reply, aware reply | illustrative copy (the SDK reads audio; it does not transcribe or generate LLM replies) |

The sample audio is synthesized (a tone shaped to the target emotion's prosody),
standing in for a real ASR front-end. The "Angry, escalating" preset drives four
real turns through the tracker, which is why it shows a genuine worsening /
escalating arc with a shift alert and a CAUTION flag.

## Files

- `index.html` — the page (vanilla JS/CSS, fonts via Google Fonts CDN).
- `generate_data.py` — runs the SDK to produce `data.js`. Re-run after SDK changes.
- `data.js` — auto-generated; do not edit by hand.
- `assets/emolog-logo-clean.png` — the wordmark logo.
