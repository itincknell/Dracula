/**
 * Presents Dracula's portrait and progressively typed narration text.
 * Portrait selection follows the public score state, while timers coordinate
 * text reveal and preserve the approved static final-loss behavior.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { HumanGameView } from "./gameView";
import type { NarrationState } from "./gameControllerContract";

export type DraculaMood = "default" | "angrier" | "angriest" | "winning";

/**
 * Every mood image is mounted at once and CSS changes only their opacity.
 * Preloading this way prevents image decode and resampling flicker during the
 * coffin animation when dialogue changes Dracula's expression.
 */
export const DRACULA_PORTRAITS: Readonly<Record<DraculaMood, string>> = {
  default: "portraits/dracula-angry-frown.jpg",
  angrier: "portraits/dracula-angrier-frown.jpg",
  angriest: "portraits/dracula-angriest-grimace.jpg",
  winning: "portraits/dracula-winning-grin.jpg",
};

export const DIALOGUE_CHARACTER_DELAY_MS = 34;

/** Select Dracula's expression from public score and round information only. */
export function draculaMood(view: HumanGameView): DraculaMood {
  // Positive difference means Dracula leads. A final loss always uses the
  // strongest angry portrait, even if it occurred before the late rounds.
  const difference = view.total_scores.opponent - view.total_scores.human;
  // A completed round's dialogue introduces the next round. Using that coming
  // round here changes the face with the dialogue, then retains it through play.
  const moodRound = view.status === "round_complete"
    ? view.round_number + 1
    : view.round_number;
  if (view.status === "game_complete" && difference < 0) return "angriest";
  if (moodRound >= 4 && difference <= -50) return "angriest";
  if (moodRound >= 4 && difference > 0) return "winning";
  if (difference < 0) return "angrier";
  return "default";
}

/**
 * Decide whether mobile layout expands a dedicated dialogue region.
 *
 * Opening text remains inside the normal portrait bar. Round-one state still
 * contains that opening message, so its transition dialogue is the second
 * message; later rounds clear their prior text and need only one. Final text
 * receives the expanded region whenever it exists.
 */
export function currentDialogueIsVisible(
  view: HumanGameView,
  messageCount: number,
): boolean {
  if (view.status === "game_complete") return messageCount > 0;
  if (view.status !== "round_complete" || view.round_number > 5) return false;
  return messageCount >= (view.round_number === 1 ? 2 : 1);
}

/** Read the user's operating-system animation preference when a browser supports it. */
function reducedMotionRequested(): boolean {
  return typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Advance the visible prefix of a line one character at a time. */
function useTypedCharacterCount(text: string, characterDelayMs: number): number {
  const [characterCount, setCharacterCount] = useState(
    reducedMotionRequested() ? text.length : 0,
  );
  useEffect(() => {
    if (reducedMotionRequested() || characterDelayMs <= 0) {
      setCharacterCount(text.length);
      return undefined;
    }
    setCharacterCount(0);
    // The interval stops itself at full length and is also removed whenever
    // React replaces the text or unmounts this dialogue.
    const timer = window.setInterval(() => {
      setCharacterCount((current) => {
        if (current >= text.length) {
          window.clearInterval(timer);
          return current;
        }
        return current + 1;
      });
    }, characterDelayMs);
    return () => window.clearInterval(timer);
  }, [characterDelayMs, text]);
  return characterCount;
}

/** Render one accessible line with a visual retro terminal typing effect. */
export function RetroDialogue({
  text,
  characterDelayMs = DIALOGUE_CHARACTER_DELAY_MS,
  onPrintStart,
}: {
  text: string;
  characterDelayMs?: number;
  onPrintStart?: () => void;
}) {
  const characterCount = useTypedCharacterCount(text, characterDelayMs);
  const startedText = useRef<string | null>(null);

  // The portrait changes on the first painted character, not when a request
  // starts or completes, so face and dialogue enter together.
  useLayoutEffect(() => {
    if (characterCount > 0 && startedText.current !== text) {
      startedText.current = text;
      onPrintStart?.();
    }
  }, [characterCount, onPrintStart, text]);

  return (
    <span className="commentary-dialogue" data-complete={characterCount >= text.length}>
      {/* Assistive technology receives the full sentence once. The animated
          duplicate and cursor are visual only, avoiding character-by-character announcements. */}
      <span className="visually-hidden">{text}</span>
      {/* Mobile reserves the complete sentence before typing starts, so its
          score card does not move downward with each newly printed line. */}
      <span className="commentary-measure" aria-hidden="true">{text}█</span>
      <span className="commentary-typed" aria-hidden="true">
        {text.slice(0, characterCount)}
        <span className="commentary-cursor">█</span>
      </span>
    </span>
  );
}

/** Keep portrait state and the text-print start event within one mounted region. */
function CommentaryContents({
  view,
  latest,
  dialogueKey,
}: {
  view: HumanGameView;
  latest: string | null;
  dialogueKey: string;
}) {
  const [mood, setMood] = useState<DraculaMood>("default");

  const beginDialogue = () => {
    setMood(draculaMood(view));
  };

  return (
    <>
      <div
        className="portrait-placeholder"
        data-mood={mood}
        aria-hidden="true"
      >
        {/* CSS displays the image named by data-mood. Keeping all images mounted
            makes a mood change an opacity update rather than a network/decode event. */}
        {(Object.entries(DRACULA_PORTRAITS) as [DraculaMood, string][]).map(
          ([portrait, path]) => (
            <img
              className="dracula-avatar"
              data-portrait={portrait}
              src={`${import.meta.env.BASE_URL}${path}`}
              alt=""
              draggable={false}
              key={portrait}
            />
          ),
        )}
      </div>
      <div
        className="commentary-stream"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        aria-relevant="additions text"
      >
        {latest === null ? null : (
          <RetroDialogue
            key={dialogueKey}
            text={latest}
            onPrintStart={beginDialogue}
          />
        )}
      </div>
    </>
  );
}

/** Render the latest successful cue and expose request/display state to CSS and ARIA. */
export function DraculaCommentary({
  view,
  narration,
}: {
  view: HumanGameView;
  narration: NarrationState;
}) {
  // The full list is intentionally tiny: it distinguishes round one's opening
  // line from its later round-transition line. Only the latest text is drawn.
  const latest = narration.messages.at(-1) ?? null;
  // Including the sequence length restarts typing when two cues happen to use
  // identical text.
  const dialogueKey = `${narration.messages.length}:${latest ?? ""}`;
  const panelRef = useRef<HTMLElement>(null);

  useLayoutEffect(() => {
    const panel = panelRef.current;
    const game = panel?.closest<HTMLElement>(".game-window");
    const stream = panel?.querySelector<HTMLElement>(".commentary-stream");
    if (!game || !stream || typeof ResizeObserver === "undefined") return;
    // The mobile text sits below the fixed portrait bar, outside the board's
    // grid. Give that grid its actual wrapped height, including after resize.
    const updateHeight = () => game.style.setProperty(
      "--round-dialogue-height", `${stream.getBoundingClientRect().height}px`,
    );
    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(stream);
    return () => {
      observer.disconnect();
      game.style.removeProperty("--round-dialogue-height");
    };
  }, []);

  return (
    <aside
      ref={panelRef}
      className="commentary-panel"
      aria-label="Dracula commentary"
      data-dialogue-visible={currentDialogueIsVisible(view, narration.messages.length)}
      aria-busy={narration.pending}
    >
      <CommentaryContents view={view} latest={latest} dialogueKey={dialogueKey} />
    </aside>
  );
}
