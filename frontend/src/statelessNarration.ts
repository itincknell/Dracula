/**
 * Coordinates the three optional narration cues around authoritative gameplay.
 * It determines cue eligibility, ignores stale asynchronous responses, and
 * keeps narration failure independent from accepted game commands.
 */
import type { HumanGameView } from "./gameView";
import type { NarrationCueType, RecoveryEnvelope } from "./statelessContracts";
import type { NarrationState } from "./gameControllerContract";
import type { StatelessApiClient } from "./statelessApi";

interface PendingNarration {
  /** Identifies the cue for the current game-history position. */
  key: string;
  /** Whether scoring has reached the point at which text may be displayed. */
  reveal: boolean;
  /** Completed text held until `reveal` becomes true, or null while unavailable. */
  text: string | null;
}

/**
 * Coordinate optional dialogue independently of authoritative gameplay.
 *
 * Opening and final dialogue may appear as soon as they arrive. A round cue is
 * requested when the server returns the completed round, held while scoring is
 * animated, and committed only after `reveal()` is called. Provider failures
 * clear narration state but never retry or alter an accepted game command.
 */
export class StatelessNarrationCoordinator {
  private pending: PendingNarration | null = null;
  // Every cancellation or new request advances the generation. A response
  // from an earlier generation is ignored even if its Promise resolves later.
  private generation = 0;

  constructor(
    private readonly api: StatelessApiClient,
    private readonly readState: () => NarrationState,
    private readonly updateState: (changes: Partial<NarrationState>) => void,
  ) {}

  /** Invalidate an outstanding request without erasing already displayed text. */
  cancel(): void {
    this.generation += 1;
    this.pending = null;
    this.updateState({ pending: false });
  }

  /** Invalidate outstanding work and remove dialogue belonging to the prior round. */
  clear(): void {
    this.generation += 1;
    this.pending = null;
    this.updateState({ pending: false, messages: [] });
  }

  /** Request the single cue allowed by the supplied settled game state. */
  requestEligible(
    view: HumanGameView,
    envelope: RecoveryEnvelope,
    { opening = false }: { opening?: boolean } = {},
  ): void {
    if (opening) {
      this.request("opening", envelope, true);
    } else if (view.status === "round_complete" && view.round_number <= 5) {
      this.request("round_transition", envelope, false);
    } else if (view.status === "game_complete") {
      this.request("final_result", envelope, true);
    }
  }

  /** Permit a completed round cue to display, immediately or when it arrives. */
  reveal(): void {
    if (this.pending === null) return;
    this.pending = { ...this.pending, reveal: true };
    if (this.pending.text !== null) this.commit(this.pending.text);
  }

  private request(
    cueType: NarrationCueType,
    envelope: RecoveryEnvelope,
    reveal: boolean,
  ): void {
    // Within one controller, history only grows and every cue point has a
    // unique history length. This key coalesces duplicate requests for the same
    // cue without introducing a second persistence or cache mechanism.
    const key = `${cueType}:${envelope.seed}:${envelope.history.length}`;
    if (this.pending?.key === key) {
      if (reveal) this.reveal();
      return;
    }
    const generation = ++this.generation;
    this.pending = { key, reveal, text: null };
    this.updateState({ pending: true });
    void this.api.narrate({ envelope, cue_type: cueType }).then((result) => {
      // The game may have been cleared or advanced while Bedrock was working.
      // Such a response belongs to an obsolete screen and must not be shown.
      if (generation !== this.generation || this.pending?.key !== key) return;
      const text = result.status === "ready" ? result.text : null;
      this.pending = { key, reveal: this.pending.reveal, text };
      this.updateState({ pending: false });
      if (this.pending.reveal && text !== null) this.commit(text);
    }).catch(() => {
      if (generation !== this.generation || this.pending?.key !== key) return;
      this.pending = null;
      this.updateState({ pending: false });
    });
  }

  /** Append one revealed cue to the observable dialogue sequence. */
  private commit(text: string): void {
    this.pending = null;
    this.updateState({
      pending: false,
      messages: [...this.readState().messages, text],
    });
  }
}
