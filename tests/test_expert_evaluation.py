"""Absolute-evaluation reporting invariants."""

from __future__ import annotations

from dracula.expert_evaluation import EvaluationGame, EvaluationRound, _game_summary


# A candidate's latency advantage must be measured against the controller it
# actually faced on the same paired games, not against an unrelated benchmark.
def test_summary_keeps_candidate_and_control_latency_separate() -> None:
    games = (
        EvaluationGame(
            "candidate-vs-search-500",
            0,
            "deck",
            "queen",
            True,
            False,
            (EvaluationRound(1, True, 20, 10),),
            (1.0,),
            (10.0,),
        ),
        EvaluationGame(
            "candidate-vs-search-500",
            0,
            "deck",
            "king",
            False,
            False,
            (EvaluationRound(1, False, 10, 20),),
            (1.2,),
            (9.0,),
        ),
    )

    summary = _game_summary("candidate-vs-search-500", games, 20)

    assert summary["decision_latency_seconds"]["mean"] == 1.1
    assert summary["control_decision_latency_seconds"]["mean"] == 9.5
    assert summary["queen"]["games"] == summary["king"]["games"] == 1
