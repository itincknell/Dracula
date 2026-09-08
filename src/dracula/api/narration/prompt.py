"""Turn allowlisted game facts into the text sent to the narrator.

The cue layer supplies structured public facts. This module expresses those
facts as a short English account so the language model can concentrate on
Dracula's voice instead of interpreting application JSON. It also supplies a
small set of example exchanges that demonstrate the desired characterization.
"""

from __future__ import annotations

from dataclasses import dataclass

from dracula.api.narration.cues import GroundedNarrationCue

# The instruction defines voice and factual limits. Round-specific content is
# kept in the user message so this text remains identical for every request.
DRACULA_SYSTEM_PROMPT = (
    "You are Count Dracula, an egotistical cartoon villain in a retro arcade "
    "card game. Speak directly to the human. Turn the supplied account into an "
    "egotistical boast, theatrical outburst, or mocking threat, not a neutral "
    "recap. Match Dracula's stated attitude: imperious is commanding, amused is "
    "playfully gloating, angry is outraged, and angrier is a grand furious "
    "outburst. Call the opponent the human or invent varied theatrical epithets "
    "such as mortal, intruder, pretender, or challenger. Vary sentence endings. "
    "Reply in roughly 20 to 42 words, usually one sentence and occasionally two. "
    "Use no profanity or modern slang. Use every important supplied fact without "
    "inventing cards, combinations, results, lead changes, or tie-breaks. Reply "
    "only with Dracula's dialogue."
)


def _sentence(value: object) -> str:
    """Normalize one generated fact fragment as a complete sentence."""

    text = str(value).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _actor(label: str, *, leading: bool = False) -> str:
    """Render a public actor label with sentence-appropriate capitalization."""

    if label == "dracula":
        return "Dracula"
    return "The human" if leading else "the human"


def _opening_summary(cue: GroundedNarrationCue) -> str:
    """Describe who opens the game and the required Dracula attitude."""

    facts = cue.facts
    human_starts = facts["first_player"] == facts["human_role"]
    starter = "The human" if human_starts else "Dracula"
    return (
        f"The game is beginning. {starter} will play first. "
        f"Dracula is {facts['dracula_attitude']}."
    )


def _standing_summary(cue: GroundedNarrationCue) -> str:
    """Describe the same score standing from one of two stable perspectives."""

    round_number = cue.facts["round"]
    leader = str(cue.facts["leader"])
    if leader == "tie":
        return f"The game is tied after round {round_number}."
    loser = "human" if leader == "dracula" else "dracula"
    if cue.wording_variant == 0:
        return f"{_actor(leader, leading=True)} is winning after round {round_number}."
    return f"{_actor(loser, leading=True)} is losing after round {round_number}."


def _round_outcome(cue: GroundedNarrationCue) -> str:
    """Describe the round result, winning combination, Vampires, and tie-break."""

    facts = cue.facts
    result = str(facts["round_result"])
    if result == "tie":
        return "The round ended in a tie."

    winner = _actor(result, leading=True)
    winner_inside_sentence = _actor(result)
    loser = "human" if result == "dracula" else "dracula"
    loser_name = _actor(loser)
    vampires = tuple(str(item) for item in facts.get("vampires_played", ()))
    loser_vampire = next(
        (item for item in vampires if item.startswith(loser_name)), None
    )
    combination = str(facts["winning_combination"])
    winning_play = (
        f"{winner_inside_sentence} won the round without a multiplier"
        if combination == "no multiplier"
        else f"{winner_inside_sentence} put together {combination}"
    )
    details = [
        _sentence(
            f"{loser_vampire}, but {winning_play}"
            if loser_vampire is not None
            else (
                f"{winner} won the round without a multiplier"
                if combination == "no multiplier"
                else f"{winner} put together {combination}"
            )
        )
    ]
    details.extend(_sentence(item) for item in vampires if item != loser_vampire)
    if "round_tie_break" in facts:
        details.append(_sentence(facts["round_tie_break"]))
    return " ".join(details)


def narration_summary(cue: GroundedNarrationCue) -> str:
    """Build the complete deterministic English account for one cue."""

    facts = cue.facts
    if cue.cue_type == "opening":
        return _opening_summary(cue)
    if cue.cue_type == "round_transition":
        return " ".join(
            (
                _standing_summary(cue),
                _sentence(facts["score_movement"]),
                _round_outcome(cue),
                f"Dracula is {facts['dracula_attitude']}.",
            )
        )
    result = str(facts["final_result"])
    outcome = (
        "The completed game ended in a draw."
        if result == "tie"
        else f"{_actor(result, leading=True)} won the completed game."
    )
    tie_break = str(facts["tie_break"]).replace("_", " ")
    return (
        f"{outcome} The result was decided by {tie_break}. "
        f"Dracula is {facts['dracula_attitude']}."
    )


# Actual prior user/assistant turns teach style without exposing formatting
# labels that the model could repeat in its answer.
DRACULA_EXAMPLE_TURNS = (
    (
        "The game is beginning. The human will play first. Dracula is imperious.",
        "Enter, little challenger, and make your opening move beneath my commanding gaze—the night already knows its rightful master!",
    ),
    (
        "Dracula is winning after round 4. Dracula extends the lead over the human. Dracula put together three Spades for a 5x multiplier. Dracula is amused.",
        "How delightful—three Spades crown me with a 5x multiplier, and my magnificent lead now stretches across the night! Struggle on, brave intruder.",
    ),
    (
        "The human is losing after round 3. Dracula takes the lead. Dracula put together two Hearts for a 2x multiplier. The players' best lines tied, and their second-best lines decided the round. Dracula is imperious.",
        "Our finest lines tied, mortal, yet my two Hearts and superior second line seized the lead—as lesser players always yield to Dracula!",
    ),
    (
        "Dracula is winning after round 2. The human catches up but still trails. Dracula played 2 vampires, but the human put together three red cards for a 3x multiplier. Dracula is imperious.",
        "Two Vampires and three red cards merely narrowed my lead? Even your darkest allies recognize Dracula as their true and eternal master!",
    ),
    (
        "Dracula is losing after round 4. The human extends the lead. The human put together two Clubs for a 2x multiplier. Dracula is angrier.",
        "Curse those two Clubs and their wretched 2x multiplier! Your swelling lead is an insult I shall repay with a century of terror!",
    ),
    (
        "Dracula won the completed game. The result was decided by total score. Dracula is amused.",
        "The final tally kneels before its master! Depart, defeated pretender, and remember forever that this table belongs to Dracula.",
    ),
)


@dataclass(frozen=True, slots=True)
class NarrationPrompt:
    """Provider text and its originating public cue."""

    system_text: str
    user_text: str
    example_turns: tuple[tuple[str, str], ...] = ()
    source_cue: GroundedNarrationCue | None = None


def build_narration_prompt(cue: GroundedNarrationCue) -> NarrationPrompt:
    """Combine the fixed voice instruction, examples, and live public facts."""

    return NarrationPrompt(
        system_text=DRACULA_SYSTEM_PROMPT,
        user_text=narration_summary(cue),
        example_turns=DRACULA_EXAMPLE_TURNS,
        source_cue=cue,
    )
