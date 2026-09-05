/**
 * Presents Dracula's portrait and progressively typed narration text.
 * Portrait selection follows the public score state, while timers coordinate
 * text reveal and preserve the approved static final-loss behavior.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { HumanGameView } from "./gameView";
import type { NarrationState } from "./gameControllerContract";

export type DraculaMood = "default" | "angrier" | "angriest" | "winning";

export const DRACULA_PORTRAITS: Readonly<Record<DraculaMood, string>> = {
  default: "portraits/dracula-angry-frown.jpg",
  angrier: "portraits/dracula-angrier-frown.jpg",
  angriest: "portraits/dracula-angriest-grimace.jpg",
  winning: "portraits/dracula-winning-grin.jpg",
};

export const DIALOGUE_CHARACTER_DELAY_MS = 34;

export function draculaMood(view: HumanGameView): DraculaMood {
  const difference = view.total_scores.opponent - view.total_scores.human;
  if (view.status === "game_complete" && difference < 0) return "angriest";
  if (view.round_number >= 4 && difference <= -50) return "angriest";
  if (view.round_number >= 4 && difference > 0) return "winning";
  if (difference < 0) return "angrier";
  return "default";
}

export function currentDialogueIsVisible(
  view: HumanGameView,
  messageCount: number,
): boolean {
  if (view.status === "game_complete") return messageCount > 0;
  if (view.status !== "round_complete" || view.round_number > 5) return false;
  return messageCount >= (view.round_number === 1 ? 2 : 1);
}

function reducedMotionRequested(): boolean {
  return typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function RetroDialogue({
  text,
  characterDelayMs = DIALOGUE_CHARACTER_DELAY_MS,
  onPrintStart,
}: {
  text: string;
  characterDelayMs?: number;
  onPrintStart?: () => void;
}) {
  const [characterCount, setCharacterCount] = useState(
    reducedMotionRequested() ? text.length : 0,
  );
  const startedText = useRef<string | null>(null);

  useEffect(() => {
    if (reducedMotionRequested() || characterDelayMs <= 0) {
      setCharacterCount(text.length);
      return undefined;
    }
    setCharacterCount(0);
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

  useLayoutEffect(() => {
    if (characterCount > 0 && startedText.current !== text) {
      startedText.current = text;
      onPrintStart?.();
    }
  }, [characterCount, onPrintStart, text]);

  return (
    <span className="commentary-dialogue" data-complete={characterCount >= text.length}>
      <span className="visually-hidden">{text}</span>
      <span aria-hidden="true">
        {text.slice(0, characterCount)}
        <span className="commentary-cursor">█</span>
      </span>
    </span>
  );
}

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

export function DraculaCommentary({
  view,
  narration,
}: {
  view: HumanGameView;
  narration: NarrationState;
}) {
  const latest = narration.messages.at(-1) ?? null;
  const dialogueKey = `${narration.messages.length}:${latest ?? ""}`;

  return (
    <aside
      className="commentary-panel"
      aria-label="Dracula commentary"
      data-narration-enabled={narration.enabled}
      data-dialogue-visible={currentDialogueIsVisible(view, narration.messages.length)}
      aria-busy={narration.pending}
    >
      <CommentaryContents view={view} latest={latest} dialogueKey={dialogueKey} />
    </aside>
  );
}
