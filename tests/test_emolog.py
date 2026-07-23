"""Test suite for emolog.

SCAFFOLD ONLY — implement the 25 tests per CLAUDE.md "tests/test_emolog.py".
Each test currently skips; remove the skip and fill in the body as you go.
"""

import numpy as np
import pytest

SR = 16000

TODO = pytest.mark.skip(reason="TODO: implement (scaffold)")


# --------------------------------------------------------------------------- #
# Suggested shared helpers (implement as needed)
# --------------------------------------------------------------------------- #
def _tone(freq=150.0, dur=3.0, amp=0.3, sr=SR):
    """Return a simple float32 mono sine tone for prosody tests."""
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_result(emotion, confidence=0.9, prosody=None):
    """Build an EmotionResult with fabricated prosody for tracker/context tests."""
    from emolog.analyzer import (
        EmotionResult,
        ProsodyFeatures,
        _estimate_arousal,
        _estimate_valence,
    )

    if prosody is None:
        prosody = ProsodyFeatures(
            speaking_rate_wps=3.0,
            energy_db=-20.0,
            energy_variance=0.0,
            pitch_mean_hz=150.0,
            pitch_variance=0.0,
            silence_ratio=0.0,
        )
    return EmotionResult(
        emotion=emotion,
        confidence=confidence,
        top3=[(emotion, confidence)],
        prosody=prosody,
        arousal=_estimate_arousal(emotion, prosody),
        valence=_estimate_valence(emotion),
    )


# --------------------------------------------------------------------------- #
class TestProsodyExtraction:
    def test_returns_prosody_features(self):
        from emolog.analyzer import ProsodyFeatures, _extract_prosody

        p = _extract_prosody(_tone(), SR, None)
        assert isinstance(p, ProsodyFeatures)
        for value in (
            p.speaking_rate_wps,
            p.energy_db,
            p.energy_variance,
            p.pitch_mean_hz,
            p.pitch_variance,
            p.silence_ratio,
        ):
            assert isinstance(value, float)

    def test_high_energy_audio(self):
        from emolog.analyzer import _extract_prosody

        loud = _extract_prosody(_tone(amp=0.5), SR, None)
        quiet = _extract_prosody(_tone(amp=0.02), SR, None)
        assert loud.energy_db > quiet.energy_db

    def test_transcription_improves_rate_estimate(self):
        from emolog.analyzer import _extract_prosody

        p = _extract_prosody(_tone(dur=3.0), SR, "one two three four five")
        assert 1.4 < p.speaking_rate_wps < 2.0  # 5 words / 3 s ≈ 1.67

    def test_silence_in_audio(self):
        from emolog.analyzer import _extract_prosody

        audio = np.concatenate(
            [_tone(dur=1.0, amp=0.3), np.zeros(int(2.0 * SR), dtype=np.float32)]
        )
        p = _extract_prosody(audio, SR, None)
        assert p.silence_ratio > 0.3


class TestArousalValence:
    def test_angry_is_high_arousal_negative_valence(self):
        from emolog.analyzer import ProsodyFeatures, _estimate_arousal, _estimate_valence

        p = ProsodyFeatures(3.0, -20.0, 0.0, 150.0, 0.0, 0.0)
        assert _estimate_arousal("angry", p) == "high"
        assert _estimate_valence("angry") == "negative"

    def test_calm_is_low_arousal_positive_valence(self):
        from emolog.analyzer import ProsodyFeatures, _estimate_arousal, _estimate_valence

        p = ProsodyFeatures(2.0, -50.0, 0.0, 120.0, 0.0, 0.0)
        assert _estimate_arousal("calm", p) == "low"
        assert _estimate_valence("calm") == "positive"

    def test_neutral_valence(self):
        from emolog.analyzer import _estimate_valence

        assert _estimate_valence("neutral") == "neutral"


class TestContextString:
    def test_contains_emotion_label(self):
        assert "angry" in _make_result("angry").context_string

    def test_contains_confidence(self):
        assert "87%" in _make_result("angry", 0.87).context_string

    def test_uncertain_emotion(self):
        assert "unclear" in _make_result("uncertain", 0.2).context_string

    def test_high_energy_description(self):
        from emolog.analyzer import ProsodyFeatures

        loud = ProsodyFeatures(3.0, -12.0, 0.0, 150.0, 0.0, 0.0)  # -12 dB -> high
        assert "high" in _make_result("angry", 0.9, loud).context_string

    def test_wrapped_in_brackets(self):
        cs = _make_result("happy").context_string
        assert cs.startswith("[Paralinguistic context:")
        assert cs.endswith("]")


class TestConversationTracker:
    def test_single_turn(self):
        from emolog.tracker import ConversationTracker

        tracker = ConversationTracker()
        state = tracker.update(_make_result("neutral"))
        assert state.turn_count == 1
        assert state.current_emotion == "neutral"
        assert state.valence_trend == "stable"
        assert state.arousal_trend == "stable"
        assert state.shift_detected is False

    def test_multi_turn_trend(self):
        from emolog.tracker import ConversationTracker

        tracker = ConversationTracker()
        for emotion in ["neutral", "neutral", "sad", "angry", "angry"]:
            state = tracker.update(_make_result(emotion))
        assert state.valence_trend == "worsening"

    def test_shift_detection(self):
        from emolog.tracker import ConversationTracker

        tracker = ConversationTracker()
        tracker.update(_make_result("neutral"))
        state = tracker.update(_make_result("angry"))
        assert state.shift_detected is True
        assert "neutral" in state.shift_description
        assert "angry" in state.shift_description

    def test_no_shift_same_emotion(self):
        from emolog.tracker import ConversationTracker

        tracker = ConversationTracker()
        tracker.update(_make_result("neutral"))
        state = tracker.update(_make_result("neutral"))
        assert state.shift_detected is False

    def test_reset_clears_history(self):
        from emolog.tracker import ConversationTracker

        tracker = ConversationTracker()
        tracker.update(_make_result("angry"))
        tracker.update(_make_result("angry"))
        tracker.reset()
        assert tracker.turn_count == 0

    def test_window_limit(self):
        from emolog.tracker import ConversationTracker

        tracker = ConversationTracker(window=3)
        for _ in range(10):
            tracker.update(_make_result("neutral"))
        assert tracker.turn_count == 3

    def test_conversation_context_string_produced(self):
        from emolog.tracker import ConversationTracker

        tracker = ConversationTracker()
        state = tracker.update(_make_result("angry"))
        cs = state.conversation_context_string
        assert cs.startswith("[Conversation emotional arc:")
        assert cs.endswith("]")


def _middleware_with_fake_analyzer(monkeypatch, emotion="angry"):
    """Build an EmologMiddleware whose analyzer is a stub, so no model loads."""
    import emolog.middleware as mw

    class _FakeAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, audio, sample_rate=SR, transcription=None):
            return _make_result(emotion)

    monkeypatch.setattr(mw, "ParalinguisticAnalyzer", _FakeAnalyzer)
    return mw.EmologMiddleware()


class TestMiddlewareInjection:
    def test_inject_adds_system_message(self, monkeypatch):
        middleware = _middleware_with_fake_analyzer(monkeypatch, "angry")
        messages = [{"role": "user", "content": "I'm fine"}]
        out = middleware.inject(np.zeros(SR, dtype=np.float32), SR, "I'm fine", messages)

        assert out[0]["role"] == "system"
        assert "Paralinguistic context" in out[0]["content"]
        # Caller's list is not mutated.
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_inject_preserves_existing_messages(self, monkeypatch):
        middleware = _middleware_with_fake_analyzer(monkeypatch, "angry")
        messages = [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "I'm fine"},
        ]
        out = middleware.inject(np.zeros(SR, dtype=np.float32), SR, "I'm fine", messages)

        systems = [m for m in out if m["role"] == "system"]
        assert len(systems) == 1
        assert "You are a helpful agent." in systems[0]["content"]
        assert "Paralinguistic context" in systems[0]["content"]
        assert any(m["role"] == "user" for m in out)

    def test_tts_style_hint_angry(self, monkeypatch):
        middleware = _middleware_with_fake_analyzer(monkeypatch, "angry")
        middleware.process(np.zeros(SR, dtype=np.float32), SR, "I'm fine")
        hint = middleware.get_tts_style_hint()

        assert float(hint["speaking_rate"]) < 1.0
        assert "empathetic" in hint["style"]
        assert all(isinstance(v, str) for v in hint.values())

    def test_tts_style_hint_no_state(self):
        from emolog.tracker import ConversationTracker
        import emolog.middleware as mw

        # No process() call -> last_conversation_state is None -> defaults.
        middleware = mw.EmologMiddleware.__new__(mw.EmologMiddleware)
        middleware.last_conversation_state = None
        hint = middleware.get_tts_style_hint()

        assert hint["style"] == "neutral"
        assert hint["speaking_rate"] == "1.0"
        assert hint["stability"] == "0.5"


class TestBenchmarkHarness:
    def test_runs_without_error(self):
        from benchmark.eval_harness import SCENARIOS, BenchmarkSuite

        def dummy_agent(transcript, emotion, context):
            return "Thanks for reaching out. Let me take a look at that."

        suite = BenchmarkSuite()
        for emolog_enabled in (True, False):
            report = suite.run(dummy_agent, emolog_enabled=emolog_enabled)
            assert len(report.results) == len(SCENARIOS) == 6
            for result in report.results:
                assert 0.0 <= result.overall <= 1.0
                assert all(0.0 <= s <= 1.0 for s in result.scores.values())
            assert 0.0 <= report.mean_score <= 1.0
            assert report.summary()
            assert report.to_json()

    def test_aware_agent_scores_higher_than_oblivious(self):
        from benchmark.eval_harness import BenchmarkSuite, support_agent

        suite = BenchmarkSuite()
        aware = suite.run(support_agent, emolog_enabled=True)
        oblivious = suite.run(support_agent, emolog_enabled=False)

        assert aware.mean_score > oblivious.mean_score

        aware_by_id = {r.scenario_id: r.overall for r in aware.results}
        oblivious_by_id = {r.scenario_id: r.overall for r in oblivious.results}
        for scenario_id in ("angry_001", "sad_001", "fearful_001"):
            assert aware_by_id[scenario_id] > oblivious_by_id[scenario_id]
