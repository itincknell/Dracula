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
  key: string;
  reveal: boolean;
  text: string | null;
}

export class StatelessNarrationCoordinator {
  private pending: PendingNarration | null = null;
  private generation = 0;

  constructor(
    private readonly api: StatelessApiClient,
    private readonly readState: () => NarrationState,
    private readonly updateState: (changes: Partial<NarrationState>) => void,
  ) {}

  cancel(): void {
    this.generation += 1;
    this.pending = null;
    this.updateState({ pending: false });
  }

  clear(): void {
    this.generation += 1;
    this.pending = null;
    this.updateState({ pending: false, messages: [] });
  }

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
    const key = `${cueType}:${envelope.seed}:${envelope.history.length}`;
    if (this.pending?.key === key) {
      if (reveal) this.reveal();
      return;
    }
    const generation = ++this.generation;
    this.pending = { key, reveal, text: null };
    this.updateState({ pending: true });
    void this.api.narrate({ envelope, cue_type: cueType }).then((result) => {
      if (generation !== this.generation || this.pending?.key !== key) return;
      const text = result.data.status === "ready" ? result.data.text : null;
      this.pending = { key, reveal: this.pending.reveal, text };
      this.updateState({ pending: false });
      if (this.pending.reveal && text !== null) this.commit(text);
    }).catch(() => {
      if (generation !== this.generation || this.pending?.key !== key) return;
      this.pending = null;
      this.updateState({ pending: false });
    });
  }

  private commit(text: string): void {
    this.pending = null;
    this.updateState({
      pending: false,
      messages: [...this.readState().messages, text],
    });
  }
}
