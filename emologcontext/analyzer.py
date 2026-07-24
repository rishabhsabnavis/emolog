"""Paralinguistic analysis: extract emotion and prosody from raw audio.

Implements CLAUDE.md "Module 1: emolog/analyzer.py", adapted to the two-tier
emotion taxonomy (EMOTION_TAXONOMY / EMOTION_LABELS / _EMOTION_AXES). Arousal
and valence are derived from the continuous _EMOTION_AXES table rather than
hardcoded per-label sets, so the operational logic stays label-agnostic as the
taxonomy grows toward a future purpose-trained model.

Constraints (see CLAUDE.md "Hard constraints"):
- No network calls anywhere.
- No disk reads inside analyze() — only during __init__ (model loading).
- Base install depends on numpy only; transformers/torch are optional extras.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Emotion taxonomy
# --------------------------------------------------------------------------- #
# Two tiers, deliberately decoupled (see design notes / CLAUDE.md deviation):
#
#   EMOTION_TAXONOMY  -> the full target label space. This is the vocabulary a
#                        future purpose-trained model is meant to emit. It is
#                        grounded in Semantic Space Theory (Cowen & Keltner's
#                        27 gradient categories) and pruned/augmented for
#                        voice-agent utility (contempt, disappointment,
#                        distress, frustration kept; purely academic states
#                        like aesthetic_appreciation dropped).
#
#   EMOTION_LABELS    -> the subset actually detectable TODAY by the shipped
#                        backends (HuBERT's IEMOCAP 4, plus what the prosody
#                        heuristic can separate). MUST be a subset of the
#                        taxonomy. Downstream code keys off _EMOTION_AXES, not
#                        this list, so growing the taxonomy never breaks the
#                        tracker / middleware / benchmark.
#
# _EMOTION_AXES maps EVERY taxonomy label to (valence, arousal, family) on
# 0..1 axes. This is the continuous SST core: labels are named peaks, but all
# operational logic runs on these two axes so unseen-but-in-taxonomy labels
# degrade gracefully to their family's coordinates.

# family -> coarse behavioral grouping used for graceful fallback.
# valence/arousal are floats in 0..1 (higher = more positive / more activated).
_EMOTION_AXES: dict[str, tuple[float, float, str]] = {
    # --- neutral / baseline -------------------------------------------------
    "neutral":        (0.50, 0.40, "neutral"),
    "calm":           (0.80, 0.20, "positive-low"),
    "contentment":    (0.85, 0.30, "positive-low"),
    "boredom":        (0.35, 0.20, "disengaged"),
    "tiredness":      (0.35, 0.15, "disengaged"),
    # --- positive / high ----------------------------------------------------
    "happy":          (1.00, 0.80, "positive-high"),
    "excited":        (0.90, 0.90, "positive-high"),
    "joy":            (1.00, 0.85, "positive-high"),
    "amusement":      (0.90, 0.70, "positive-high"),
    "relief":         (0.75, 0.40, "positive-low"),
    "pride":          (0.85, 0.65, "positive-high"),
    "gratitude":      (0.90, 0.50, "positive-low"),
    "interest":       (0.70, 0.60, "engaged"),
    # --- surprise (valence-ambiguous) --------------------------------------
    "surprised":      (0.50, 0.70, "surprise"),
    "confused":       (0.40, 0.50, "surprise"),
    "doubt":          (0.40, 0.45, "surprise"),
    # --- negative / high (activated distress) ------------------------------
    "angry":          (0.00, 1.00, "negative-high"),
    "frustrated":     (0.15, 0.80, "negative-high"),
    "fearful":        (0.20, 0.90, "negative-high"),
    "distressed":     (0.15, 0.85, "negative-high"),
    "disgusted":      (0.00, 0.60, "negative-high"),
    "contempt":       (0.10, 0.55, "negative-high"),
    "anxious":        (0.25, 0.75, "negative-high"),
    # --- negative / low (deactivated distress) -----------------------------
    "sad":            (0.10, 0.20, "negative-low"),
    "disappointed":   (0.20, 0.35, "negative-low"),
    "embarrassed":    (0.25, 0.55, "negative-low"),
    "guilt":          (0.20, 0.45, "negative-low"),
    "shame":          (0.15, 0.40, "negative-low"),
    # --- uncertain sentinel (below confidence_threshold) -------------------
    "uncertain":      (0.50, 0.40, "neutral"),
}

# Full target vocabulary (excludes the "uncertain" runtime sentinel).
EMOTION_TAXONOMY = [k for k in _EMOTION_AXES if k != "uncertain"]

# Detectable TODAY by shipped backends. MUST stay a subset of EMOTION_TAXONOMY.
# (HuBERT/IEMOCAP: neutral/happy/sad/angry; heuristic adds fearful/calm/
# disgusted/surprised/contempt.) Kept spec-compatible order for the first 8.
EMOTION_LABELS = [
    "neutral", "happy", "sad", "angry",
    "fearful", "disgusted", "surprised", "calm",
    "contempt",
]

assert set(EMOTION_LABELS).issubset(EMOTION_TAXONOMY), (
    "EMOTION_LABELS must be a subset of EMOTION_TAXONOMY"
)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class ProsodyFeatures:
    """Frame-level prosodic summary of an utterance. All fields are floats."""

    speaking_rate_wps: float      # estimated words per second
    energy_db: float              # RMS energy in decibels
    energy_variance: float        # variance of frame-level RMS (emphasis/stress)
    pitch_mean_hz: float          # mean fundamental frequency estimate
    pitch_variance: float         # variance of F0 (intonation range)
    silence_ratio: float          # fraction of frames below energy threshold (0..1)


@dataclass
class EmotionResult:
    """Primary output of ParalinguisticAnalyzer.analyze()."""

    emotion: str
    confidence: float
    top3: list[tuple[str, float]]
    prosody: ProsodyFeatures
    arousal: Literal["high", "medium", "low"]
    valence: Literal["positive", "neutral", "negative"]
    # context_string is computed in __post_init__, NOT a constructor arg.
    context_string: str = field(init=False, default="")
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        self.context_string = _build_context_string(self)


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
class ParalinguisticAnalyzer:
    def __init__(
        self,
        backend: Literal["hubert", "whisper"] = "hubert",
        model_name: str = "superb/hubert-base-superb-er",
        device: Optional[str] = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device if device is not None else self._auto_device()
        self._load_model()

    @staticmethod
    def _auto_device() -> str:
        """Return 'cuda' if torch reports a GPU, else 'cpu'. Never fails if
        torch is absent (heuristic backend needs only numpy)."""
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # -- model loading ----------------------------------------------------- #
    def _load_model(self) -> None:
        if self.backend == "hubert":
            self._load_hubert()
        elif self.backend == "whisper":
            self._load_whisper()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _load_hubert(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError as exc:
            raise ImportError(
                "HuBERT backend requires: pip install transformers torch torchaudio"
            ) from exc

        self._processor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self._model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _load_whisper(self) -> None:
        """The "whisper" backend classifies via a pure-prosody heuristic that
        needs only numpy. Whisper itself is loaded lazily *only if installed*,
        for optional on-device transcription — its absence is NOT fatal, so
        `pip install emologcontext` (numpy only) + backend="whisper" works out of the
        box. (Deliberate deviation from the spec's hard ImportError, to honor
        the README's "heuristic backend runs on the base install" promise.)"""
        self._whisper_model = None
        try:
            import whisper

            self._whisper_model = whisper.load_model("base")
        except ImportError:
            self._whisper_model = None

    # -- public API -------------------------------------------------------- #
    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        transcription: Optional[str] = None,
    ) -> EmotionResult:
        start = time.perf_counter()

        audio = _ensure_mono_float32(audio, sample_rate)
        prosody = _extract_prosody(audio, sample_rate, transcription)

        if self.backend == "hubert":
            emotion, confidence, top3 = self._classify_hubert(audio, sample_rate)
        else:
            emotion, confidence, top3 = self._classify_whisper_heuristic(audio, prosody)

        if confidence < self.confidence_threshold:
            emotion = "uncertain"

        arousal = _estimate_arousal(emotion, prosody)
        valence = _estimate_valence(emotion)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            top3=top3,
            prosody=prosody,
            arousal=arousal,
            valence=valence,
            latency_ms=latency_ms,
        )

    # -- classifiers ------------------------------------------------------- #
    def _classify_hubert(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[str, float, list[tuple[str, float]]]:
        import torch

        inputs = self._processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        id2label = self._model.config.id2label
        pairs = [(id2label[i].lower(), float(probs[i])) for i in range(len(probs))]
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        top_label, top_prob = pairs[0]
        return top_label, top_prob, pairs[:3]

    def _classify_whisper_heuristic(
        self, audio: np.ndarray, prosody: ProsodyFeatures
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Pure heuristic — no model call. Estimate emotion from prosody only."""
        return _heuristic_from_prosody(prosody)


# --------------------------------------------------------------------------- #
# Prosody-only heuristic classifier
# --------------------------------------------------------------------------- #
# Each emotion is a prototype point in a normalized (energy, rate, pitch,
# emphasis) feature cube; classification is nearest-prototype by weighted
# Euclidean distance. This replaces an earlier set of hand-tuned mean() rules
# that had two structural blind spots worth recording, since they motivate the
# design here:
#
#   1. pitch_variance is estimated from zero-crossing-rate fractions, whose
#      variance sits around 1e-5..1e-4 — far below the /0.02 scale the old rules
#      normalized it by. So the "pv" term was effectively always ~0, which
#      handed `calm` a free ~1.0 term and let it swallow ALL quiet/slow audio.
#      `sad` (quiet, slow, LOW-pitched) could never outrank `calm` because they
#      shared two terms and calm's third was always larger. pitch_variance is
#      therefore dropped as a primary feature; the honest separator between sad
#      and calm is absolute pitch (sadness sits lower), which prototypes capture.
#   2. `neutral` was a flat 0.45 baseline that only won when every other label
#      fell below it — a regime that requires high energy+rate, which is exactly
#      the angry/happy basin. So `neutral` was unreachable too. Here it is a
#      real prototype (moderate everything, low emphasis) with its own basin.
#
# energy_variance ("emphasis") is added as a 4th axis so `angry` (emphatic
# bursts) separates cleanly from `happy`/`neutral` at similar loudness.
#
# Prototype coordinates: (energy, rate, pitch, emphasis), each 0..1.
_HEURISTIC_PROTOTYPES: dict[str, tuple[float, float, float, float]] = {
    "neutral":   (0.58, 0.52, 0.45, 0.18),
    "angry":     (0.95, 0.75, 0.55, 0.85),
    "happy":     (0.85, 0.70, 0.80, 0.45),
    "sad":       (0.22, 0.28, 0.28, 0.15),
    "calm":      (0.40, 0.35, 0.48, 0.15),
    "fearful":   (0.72, 0.75, 0.75, 0.55),
    "surprised": (0.65, 0.55, 0.90, 0.45),
    "disgusted": (0.45, 0.32, 0.32, 0.45),
    "contempt":  (0.42, 0.40, 0.42, 0.30),
}
# Per-axis trust: pitch and emphasis are noisier proxies than energy/rate.
_HEURISTIC_WEIGHTS = (1.1, 1.0, 0.9, 0.7)


def _heuristic_features(prosody: ProsodyFeatures) -> tuple[float, ...]:
    """Project prosody onto the normalized (energy, rate, pitch, emphasis) cube
    the prototypes live in."""
    return (
        _clip01((prosody.energy_db + 55.0) / 45.0),   # -55 dB -> 0, -10 dB -> 1
        _clip01(prosody.speaking_rate_wps / 5.0),     # 0 -> 0, 5 wps -> 1
        _clip01(prosody.pitch_mean_hz / 300.0),       # 0 -> 0, 300 Hz -> 1
        _clip01(prosody.energy_variance / 0.004),     # emphasis / stress
    )


def _heuristic_from_prosody(
    prosody: ProsodyFeatures,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Nearest-prototype classification over the heuristic label set. Prosody is
    a weak signal for subtler emotions (disgust/contempt), so genuinely
    ambiguous audio stays flat and gets demoted to "uncertain" upstream by the
    confidence threshold — which is the honest behavior."""
    x = _heuristic_features(prosody)
    w = _HEURISTIC_WEIGHTS
    w_norm = math.sqrt(sum(w))

    # Similarity = 1 - normalized weighted distance to each prototype.
    scores = {}
    for label, proto in _HEURISTIC_PROTOTYPES.items():
        dist = math.sqrt(sum(wi * (xi - pi) ** 2 for wi, xi, pi in zip(w, x, proto)))
        scores[label] = max(0.0, 1.0 - dist / w_norm)

    total = sum(scores.values())
    dist_probs = {k: v / total for k, v in scores.items()}
    # Sharpen (power transform, gamma>1) so a clear prosodic winner earns a
    # usable confidence, while ambiguous audio stays flat. Monotonic — preserves
    # the ranking.
    gamma = 9.0
    powered = {k: v ** gamma for k, v in dist_probs.items()}
    z = sum(powered.values())
    ranked = sorted(
        ((k, v / z) for k, v in powered.items()), key=lambda kv: kv[1], reverse=True
    )
    top_label, top_prob = ranked[0]
    return top_label, top_prob, ranked[:3]


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _frame_signal(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Slice `audio` into overlapping frames -> 2D array (n_frames, frame_len).
    Pads to at least one full frame."""
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    n_frames = 1 + (len(audio) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return audio[idx]


def _extract_prosody(
    audio: np.ndarray, sr: int, transcription: Optional[str]
) -> ProsodyFeatures:
    """25ms frames / 10ms hop. Compute energy (dB + variance), silence ratio,
    pitch via zero-crossing-rate proxy, and speaking rate."""
    audio = np.asarray(audio, dtype=np.float32)
    duration_s = max(len(audio) / sr, 1e-6)

    frame_len = max(1, int(0.025 * sr))
    hop = max(1, int(0.010 * sr))
    frames = _frame_signal(audio, frame_len, hop)

    # --- energy ---------------------------------------------------------- #
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    energy_db = float(20.0 * np.log10(float(np.mean(rms)) + 1e-9))
    energy_variance = float(np.var(rms))

    # --- silence --------------------------------------------------------- #
    # Silence is defined RELATIVE to the loudest frames (a fraction of peak
    # RMS), so genuine digital silence is caught while a steady-energy signal
    # is not spuriously flagged as all-silence (which a self-referential
    # percentile threshold would do on near-constant audio).
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    threshold = 0.15 * peak_rms
    voiced_mask = rms > threshold
    silence_ratio = float(np.mean(~voiced_mask))

    # --- pitch (zero-crossing-rate proxy on voiced frames) --------------- #
    signs = np.sign(frames)
    crossings = np.abs(np.diff(signs, axis=1)) > 0     # bool per adjacent pair
    zcr = np.mean(crossings, axis=1)                    # fraction of samples
    if np.any(voiced_mask):
        voiced_zcr = zcr[voiced_mask]
        pitch_mean_hz = float(np.mean(voiced_zcr) * sr / 2.0)
        pitch_variance = float(np.var(voiced_zcr))
    else:
        pitch_mean_hz = 0.0
        pitch_variance = 0.0

    # --- speaking rate --------------------------------------------------- #
    if transcription:
        speaking_rate_wps = float(len(transcription.split()) / duration_s)
    else:
        speaking_rate_wps = _syllable_rate(rms, hop, sr, duration_s)

    return ProsodyFeatures(
        speaking_rate_wps=speaking_rate_wps,
        energy_db=energy_db,
        energy_variance=energy_variance,
        pitch_mean_hz=pitch_mean_hz,
        pitch_variance=pitch_variance,
        silence_ratio=silence_ratio,
    )


def _syllable_rate(rms: np.ndarray, hop: int, sr: int, duration_s: float) -> float:
    """Estimate words/sec by counting syllable-like peaks in the smoothed RMS
    envelope (each ~2 syllables ≈ 1 word is close enough for a proxy)."""
    if len(rms) < 3:
        return 0.0
    k = max(1, int(0.05 * sr / hop))            # ~50 ms smoothing window
    kernel = np.ones(k) / k
    smooth = np.convolve(rms, kernel, mode="same")
    thr = float(np.mean(smooth))
    mid, prev, nxt = smooth[1:-1], smooth[:-2], smooth[2:]
    peaks = int(np.sum((mid > prev) & (mid >= nxt) & (mid > thr)))
    return float(peaks / max(duration_s, 1e-6))


def _ensure_mono_float32(audio: np.ndarray, sr: int) -> np.ndarray:
    """Cast to float32; average channels if stereo; normalize if max abs > 1.0."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def _estimate_arousal(
    emotion: str, prosody: ProsodyFeatures
) -> Literal["high", "medium", "low"]:
    """Bucket the emotion's continuous arousal coordinate; fall back to prosody
    for 'uncertain'/unknown labels."""
    axes = _EMOTION_AXES.get(emotion)
    if axes is not None and emotion != "uncertain":
        arousal = axes[1]
        if arousal >= 0.6:
            return "high"
        if arousal <= 0.4:
            return "low"
        return "medium"
    # Unknown / uncertain -> infer directly from prosody.
    if prosody.energy_db > -25.0 and prosody.speaking_rate_wps > 3.5:
        return "high"
    if prosody.energy_db < -45.0:
        return "low"
    return "medium"


def _estimate_valence(emotion: str) -> Literal["positive", "neutral", "negative"]:
    """Bucket the emotion's continuous valence coordinate."""
    axes = _EMOTION_AXES.get(emotion)
    if axes is None or emotion == "uncertain":
        return "neutral"
    valence = axes[0]
    if valence > 0.6:
        return "positive"
    if valence < 0.4:
        return "negative"
    return "neutral"


def _build_context_string(result: EmotionResult) -> str:
    """Injection-ready string wrapped in '[Paralinguistic context: ... ]'."""
    p = result.prosody
    parts: list[str] = []

    if result.emotion == "uncertain":
        parts.append("The user's emotional state is unclear from this audio.")
    else:
        parts.append(
            f"The user sounds {result.emotion} "
            f"({result.confidence * 100:.0f}% confidence)."
        )

    # Speaking rate (wps).
    rate = p.speaking_rate_wps
    if rate < 1.5:
        rate_desc = "very slow"
    elif rate < 2.5:
        rate_desc = "slow"
    elif rate < 3.5:
        rate_desc = "moderate"
    elif rate < 4.5:
        rate_desc = "fast"
    else:
        rate_desc = "very fast"
    parts.append(f"Speaking rate is {rate_desc}.")

    # Vocal energy (dB).
    energy = p.energy_db
    if energy < -45.0:
        energy_desc = "low"
    elif energy < -25.0:
        energy_desc = "moderate"
    elif energy < -10.0:
        energy_desc = "high"
    else:
        energy_desc = "very high"
    parts.append(f"Vocal energy is {energy_desc}.")

    if p.energy_variance > 0.002:
        parts.append("There is notable emphasis and stress in the delivery.")
    if p.pitch_variance > 0.01:
        parts.append("Intonation is expressive, with a wide pitch range.")
    if p.silence_ratio > 0.5:
        parts.append("There is noticeable hesitation, with frequent pauses.")

    return "[Paralinguistic context: " + " ".join(parts) + " ]"
