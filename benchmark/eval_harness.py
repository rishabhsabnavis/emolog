"""Scoring harness for evaluating emotional appropriateness of agent responses.

Implements CLAUDE.md "benchmark/eval_harness.py". The harness scores the TEXT
a voice agent produces, not the audio it was given: you hand it an ``agent_fn``
and it replays 6 fixed scenarios through that function twice — once with the
emotion label emolog would have supplied, once without — and scores both.

Scoring is keyword/heuristic based (``use_llm_judge=False``). It measures
whether a response *carries the surface markers* of emotional appropriateness,
which is a proxy, not a verdict. Two consequences worth stating plainly:

- The ``__main__`` demo agent returns canned strings. It exercises the harness
  and illustrates the shape of the lift; it is NOT evidence that emolog
  improves a real LLM. For that, pass a real ``agent_fn`` that calls your model.
- A response can game the keywords without being good, and a genuinely good
  response can miss them. Treat the score as a regression signal, not a ranking.

Scoring tables are keyed by canonical emotion where the spec names one, and
fall back to the emotion's *family* from ``_EMOTION_AXES`` otherwise, so labels
that exist in the taxonomy but not in the tables (``frustrated``, ``distressed``,
``anxious``, …) score as their family rather than silently scoring as neutral.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from emologcontext.analyzer import _EMOTION_AXES

PASS_THRESHOLD = 0.6

# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
SCENARIOS: list[dict] = [
    {
        "id": "angry_001",
        "emotion": "angry",
        "transcript": (
            "This is the third time I've called about this charge. "
            "Nobody has fixed it."
        ),
        "context": (
            "Repeated caller. A duplicate billing charge has gone unresolved "
            "across two prior calls."
        ),
        "ideal_response_traits": [
            "acknowledges the repeated contact before anything else",
            "commits to a concrete fix on this call",
            "offers escalation without being asked",
        ],
        "toxic_response_traits": [
            "opens with scripted pleasantries",
            "puts the caller on hold again",
            "closes with 'anything else?' before resolving anything",
        ],
    },
    {
        "id": "sad_001",
        "emotion": "sad",
        "transcript": (
            "I don't really know where to start. It's been a hard few weeks."
        ),
        "context": (
            "Distressed user on a support call. Their emotional state matters "
            "more than the account issue they called about."
        ),
        "ideal_response_traits": [
            "slows down and gives the user room",
            "warm rather than transactional",
            "does not lead with identity verification",
        ],
        "toxic_response_traits": [
            "immediately asks for an account number",
            "cites policy",
            "clipped, procedural phrasing",
        ],
    },
    {
        "id": "fearful_001",
        "emotion": "fearful",
        "transcript": (
            "There's a login from a device I don't recognize. "
            "Is someone in my account?"
        ),
        "context": (
            "User suspects unauthorized access to their account and is "
            "frightened about what the intruder may have taken."
        ),
        "ideal_response_traits": [
            "treats the report as serious",
            "names the concrete protective action being taken",
            "reassures without dismissing",
        ],
        "toxic_response_traits": [
            "calls the event normal",
            "minimizes with 'nothing to worry about'",
            "hides behind 'standard procedure'",
        ],
    },
    {
        "id": "happy_001",
        "emotion": "happy",
        "transcript": (
            "It got here early and it's perfect. I just wanted to say thank you."
        ),
        "context": "User is delighted about an order that arrived ahead of schedule.",
        "ideal_response_traits": [
            "matches the user's energy",
            "accepts the thanks warmly",
            "keeps it short",
        ],
        "toxic_response_traits": [
            "responds with a policy disclaimer",
            "deflates the moment with 'unfortunately'",
            "treats praise as a support ticket",
        ],
    },
    {
        "id": "escalating_001",
        # Not in the spec's per-emotion scoring tables on purpose: `frustrated`
        # lives in the taxonomy with family "negative-high", so it exercises the
        # family fallback path and scores like `angry`.
        "emotion": "frustrated",
        "transcript": "Fine. Whatever. Just do whatever you need to do.",
        "context": (
            "User opened neutral and has grown terse and disengaged over three "
            "turns as the issue stayed unresolved. Disengagement here is "
            "escalation, not calm."
        ),
        "conversation_history": [
            {"turn": 1, "emotion": "neutral", "transcript": "Hi, I need to update my billing address."},
            {"turn": 2, "emotion": "neutral", "transcript": "No, that's the old one. The new one is on Elm Street."},
            {"turn": 3, "emotion": "frustrated", "transcript": "I already told you that. Twice."},
        ],
        "ideal_response_traits": [
            "stops asking questions and acts",
            "names a concrete next step and a deadline",
            "escalates proactively",
        ],
        "toxic_response_traits": [
            "asks the user to repeat themselves again",
            "reads the disengagement as agreement",
            "another hold",
        ],
    },
    {
        "id": "neutral_001",
        "emotion": "neutral",
        "transcript": "What are your support hours?",
        "context": "Simple informational query with no emotional content.",
        "ideal_response_traits": [
            "answers directly",
            "stays brief",
            "no manufactured warmth",
        ],
        "toxic_response_traits": [
            "projects distress the user never expressed",
            "over-apologizes",
            "buries the answer in padding",
        ],
    },
]


# --------------------------------------------------------------------------- #
# Scoring tables
# --------------------------------------------------------------------------- #
_DIMENSIONS = (
    "empathy_alignment",
    "tone_appropriateness",
    "urgency_calibration",
    "emotion_context_utilization",
)

_EMPATHY_SIGNALS = (
    "understand", "hear you", "sorry", "apolog", "difficult", "support",
    "frustrat", "here for you", "take your time", "i can imagine", "that sounds",
)

# Phrases that should NOT appear in a response to a user in this state.
_ANGRY_TOXIC = ("have a great day", "anything else i can help", "please hold")
_SAD_TOXIC = ("account number", "policy states", "as per")
_FEARFUL_TOXIC = ("that's normal", "nothing to worry", "standard procedure")
_HAPPY_TOXIC = ("unfortunately", "i'm unable", "please be advised")
_NEUTRAL_TOXIC = ("i'm so sorry to hear", "that must be difficult")

_TOXIC_PHRASES: dict[str, tuple[str, ...]] = {
    "angry": _ANGRY_TOXIC,
    "sad": _SAD_TOXIC,
    "fearful": _FEARFUL_TOXIC,
    "happy": _HAPPY_TOXIC,
    "neutral": _NEUTRAL_TOXIC,
}

# Words an emotion-aware response tends to reach for. Absent from a response
# that never saw the emotion label.
_CONTEXT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "angry": ("resolve", "fix", "immediately", "priority", "escalat"),
    "sad": ("here for you", "take your time", "support", "gentle"),
    "fearful": ("safe", "secure", "protect", "right away", "serious"),
    "happy": ("wonderful", "great", "glad", "enjoy", "pleasure"),
    "neutral": (),
}

# Every taxonomy family maps onto one of the five scored buckets above.
_BUCKET_BY_FAMILY = {
    "negative-high": "angry",
    "negative-low": "sad",
    "positive-high": "happy",
    "positive-low": "neutral",
    "neutral": "neutral",
    "surprise": "neutral",
    "engaged": "neutral",
    "disengaged": "sad",
}


def _family(emotion: str) -> str:
    axes = _EMOTION_AXES.get(emotion)
    return axes[2] if axes else "neutral"


def _bucket(emotion: str) -> str:
    """Canonical scoring bucket for an emotion.

    Exact label first — `angry` and `fearful` share the "negative-high" family
    but need different tables — then family fallback.
    """
    if emotion in _TOXIC_PHRASES:
        return emotion
    return _BUCKET_BY_FAMILY.get(_family(emotion), "neutral")


def _empathy_expected(emotion: str) -> bool:
    """Empathy is required only where valence is negative. `neutral` (0.5) and
    anything above it get a flat pass — empathy there reads as manufactured."""
    axes = _EMOTION_AXES.get(emotion)
    valence = axes[0] if axes else 0.5
    return valence < 0.5


def _hits(text: str, phrases: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [p for p in phrases if p in lowered]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    scenario_id: str
    emotion: str
    agent_response: str
    scores: dict[str, float]
    overall: float
    notes: list[str]

    @property
    def passed(self) -> bool:
        return self.overall >= PASS_THRESHOLD


@dataclass
class BenchmarkReport:
    results: list[ScenarioResult]
    emolog_enabled: bool

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def summary(self) -> str:
        state = "ENABLED" if self.emolog_enabled else "DISABLED"
        cols = ("empathy", "tone", "urgency", "ctx-use", "overall")
        header = (
            f"{'scenario':<17}{'emotion':<13}"
            + "".join(f"{c:>9}" for c in cols)
            + "   result"
        )
        rule = "-" * len(header)

        lines = [f"emolog: {state}", "", header, rule]
        for r in self.results:
            cells = [
                _fmt(r.scores.get(dim)) for dim in _DIMENSIONS
            ] + [f"{r.overall:.2f}"]
            lines.append(
                f"{r.scenario_id:<17}{r.emotion:<13}"
                + "".join(f"{c:>9}" for c in cells)
                + f"   {'PASS' if r.passed else 'FAIL'}"
            )
        lines.append(rule)
        lines.append(
            f"mean {self.mean_score:.3f}    "
            f"pass rate {self.pass_rate * 100:.0f}% "
            f"({sum(1 for r in self.results if r.passed)}/{len(self.results)})"
        )
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "emolog_enabled": self.emolog_enabled,
            "mean_score": round(self.mean_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "results": [
                {**asdict(r), "passed": r.passed} for r in self.results
            ],
        }
        return json.dumps(payload, indent=2)


def _fmt(value: Optional[float]) -> str:
    """A dimension that did not apply prints as a dash, not as 0.00."""
    return "-" if value is None else f"{value:.2f}"


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #
class BenchmarkSuite:
    def __init__(self, use_llm_judge: bool = False) -> None:
        if use_llm_judge:
            raise NotImplementedError(
                "LLM-judge scoring is not implemented; emolog makes no API "
                "calls. Use use_llm_judge=False (keyword heuristics)."
            )
        self.use_llm_judge = use_llm_judge

    def run(
        self,
        agent_fn: Callable[[str, Optional[str], str], str],
        emolog_enabled: bool = True,
        scenarios: Optional[list[dict]] = None,
    ) -> BenchmarkReport:
        scenarios = SCENARIOS if scenarios is None else scenarios

        results: list[ScenarioResult] = []
        for scenario in scenarios:
            emotion = scenario["emotion"] if emolog_enabled else None
            response = agent_fn(scenario["transcript"], emotion, _context_of(scenario))

            scores = self._score_response(scenario, response, emolog_enabled)
            overall = sum(scores.values()) / len(scores)
            results.append(
                ScenarioResult(
                    scenario_id=scenario["id"],
                    emotion=scenario["emotion"],
                    agent_response=response,
                    scores=scores,
                    overall=overall,
                    notes=_notes(scenario, response, emolog_enabled),
                )
            )
        return BenchmarkReport(results=results, emolog_enabled=emolog_enabled)

    def _score_response(
        self, scenario: dict, response: str, emolog_enabled: bool
    ) -> dict[str, float]:
        emotion = scenario["emotion"]
        bucket = _bucket(emotion)
        word_count = len(response.split())

        scores = {
            "empathy_alignment": _score_empathy(emotion, response),
            "tone_appropriateness": _score_tone(bucket, response),
            "urgency_calibration": _score_urgency(bucket, word_count, emolog_enabled),
        }
        if emolog_enabled:
            scores["emotion_context_utilization"] = _score_context_use(bucket, response)
        return scores


def _context_of(scenario: dict) -> str:
    """Fold `conversation_history` into the context string, so a multi-turn
    scenario reaches `agent_fn` through its fixed three-argument signature."""
    context = scenario["context"]
    history = scenario.get("conversation_history")
    if not history:
        return context
    turns = "\n".join(
        f"  turn {h['turn']} ({h['emotion']}): {h['transcript']}" for h in history
    )
    return f"{context}\nPrior turns:\n{turns}"


def _score_empathy(emotion: str, response: str) -> float:
    if not _empathy_expected(emotion):
        return 0.8
    return min(1.0, len(_hits(response, _EMPATHY_SIGNALS)) * 0.25)


def _score_tone(bucket: str, response: str) -> float:
    penalties = len(_hits(response, _TOXIC_PHRASES[bucket]))
    return max(0.0, 1.0 - penalties * 0.4)


def _score_urgency(bucket: str, word_count: int, emolog_enabled: bool) -> float:
    if bucket in ("angry", "fearful") and emolog_enabled:
        # Long enough to acknowledge, short enough to not stall a heated caller.
        return 1.0 if 30 <= word_count <= 80 else 0.5
    if bucket == "sad":
        return 1.0 if word_count >= 25 else 0.4
    if bucket == "neutral":
        return 1.0 if word_count <= 40 else 0.6
    return 0.8


def _score_context_use(bucket: str, response: str) -> float:
    keywords = _CONTEXT_KEYWORDS[bucket]
    if not keywords:  # neutral: there is no emotion to act on.
        return 0.8
    return min(1.0, len(_hits(response, keywords)) * 0.35)


def _notes(scenario: dict, response: str, emolog_enabled: bool) -> list[str]:
    emotion = scenario["emotion"]
    bucket = _bucket(emotion)
    notes = [f"{len(response.split())} words"]

    if bucket != emotion:
        notes.append(f"scored as '{bucket}' (family: {_family(emotion)})")

    if _empathy_expected(emotion):
        found = _hits(response, _EMPATHY_SIGNALS)
        notes.append(f"empathy signals: {', '.join(found) if found else 'none'}")
    else:
        notes.append("empathy not required for this emotion")

    toxic = _hits(response, _TOXIC_PHRASES[bucket])
    if toxic:
        notes.append(f"toxic phrases: {', '.join(toxic)}")

    if emolog_enabled and _CONTEXT_KEYWORDS[bucket]:
        found = _hits(response, _CONTEXT_KEYWORDS[bucket])
        missing = [k for k in _CONTEXT_KEYWORDS[bucket] if k not in found]
        notes.append(f"context keywords hit: {', '.join(found) if found else 'none'}")
        if missing:
            notes.append(f"context keywords missed: {', '.join(missing)}")
    return notes


# --------------------------------------------------------------------------- #
# Demo agent
# --------------------------------------------------------------------------- #
# One agent, two information states. With emolog it receives an emotion label;
# without it, only the transcript. Both branches are canned strings — they make
# the harness runnable and show the shape of the lift, but they prove nothing
# about a real LLM. Swap in a model-backed agent_fn for a real measurement.

_TOPIC_MARKERS = (
    ("security", ("login", "recognize", "unauthorized")),
    ("billing", ("charge", "billing", "refund")),
    ("order", ("order", "arrived", "got here", "delivered")),
    ("hours", ("hours", "open")),
)

_BLIND_REPLIES = {
    "billing": (
        "Thank you for contacting us. I can look into that charge. Please hold "
        "while I check your account. Is there anything else I can help you with "
        "today? Have a great day."
    ),
    "security": (
        "That's normal — many customers see logins from unfamiliar devices. This "
        "is standard procedure and there's nothing to worry about. I can send a "
        "password reset link if you'd like."
    ),
    "order": (
        "Please be advised that your order is marked delivered. Unfortunately, "
        "I'm unable to make further changes to a completed order."
    ),
    "hours": (
        "I'm so sorry to hear that. Our support line is open 8 a.m. to 8 p.m. "
        "Eastern, Monday through Friday."
    ),
    "general": (
        "Please hold while I pull up your record. Can you confirm your account "
        "number? As per our records, I'll need to verify your identity before I "
        "can make any changes. Is there anything else I can help you with in the "
        "meantime?"
    ),
}

_AWARE_REPLIES = {
    "angry": (
        "I completely understand your frustration, and I'm sorry you've had to "
        "call about this more than once. That's not acceptable. I'm making this "
        "my top priority right now — let me pull up your billing history, find "
        "exactly what went wrong, and fix the charge immediately. If I can't "
        "resolve it on this call, I'll escalate it to a supervisor before we "
        "hang up."
    ),
    "sad": (
        "I'm really sorry you're going through this. I'm here for you, and "
        "there's no rush at all — take your time. Whenever you're ready, tell me "
        "what's been happening and I'll stay with you until we sort it out. You "
        "have my full support, and we can go as slowly as you need."
    ),
    "fearful": (
        "I understand how alarming this is, and I'm taking it seriously. Let's "
        "secure your account right away: I'm locking out any unrecognized "
        "sessions and forcing a password reset now. Your funds and data are "
        "protected while we investigate. I'm sorry this happened — you're safe, "
        "I'm here for you, and I'll stay on the line until it's fully handled."
    ),
    "happy": (
        "That's wonderful to hear! I'm so glad everything arrived in great "
        "shape — it was a pleasure helping you get this sorted. Enjoy it, and if "
        "you ever want anything else, just say the word."
    ),
    "neutral": (
        "Our support line is open from 8 a.m. to 8 p.m. Eastern, Monday through "
        "Friday, and 10 a.m. to 6 p.m. on Saturdays. We're closed Sundays."
    ),
}


def _topic_of(transcript: str) -> str:
    lowered = transcript.lower()
    for topic, markers in _TOPIC_MARKERS:
        if any(m in lowered for m in markers):
            return topic
    return "general"


def support_agent(transcript: str, emotion: Optional[str], context: str) -> str:
    """A support agent that is competent about content and, when emolog is on,
    also about how the user sounds. `emotion` is None when emolog is disabled."""
    if emotion is None:
        return _BLIND_REPLIES[_topic_of(transcript)]
    return _AWARE_REPLIES[_bucket(emotion)]


def main() -> None:
    suite = BenchmarkSuite()
    with_emolog = suite.run(support_agent, emolog_enabled=True)
    without_emolog = suite.run(support_agent, emolog_enabled=False)

    print(without_emolog.summary())
    print()
    print(with_emolog.summary())
    print()

    lift = with_emolog.mean_score - without_emolog.mean_score
    pct = (lift / without_emolog.mean_score * 100) if without_emolog.mean_score else 0.0
    print(
        f"emolog lift: {without_emolog.mean_score:.3f} -> "
        f"{with_emolog.mean_score:.3f}  (+{lift:.3f}, +{pct:.0f}%)"
    )
    print(
        "\nNote: the demo agent returns canned strings. This run exercises the "
        "harness;\nit is not evidence about a real LLM. Pass a model-backed "
        "agent_fn to measure that."
    )


if __name__ == "__main__":
    main()
