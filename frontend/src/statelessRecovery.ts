/**
 * Persists the minimal stateless recovery envelope in browser storage.
 * Only the visible seed and accepted command history are retained; malformed
 * stored data is discarded rather than passed to gameplay reconstruction.
 */
import { assertRecoveryEnvelope, type RecoveryEnvelope } from "./statelessContracts";

export const RECOVERY_STORAGE_KEY = "dracula.recovery-envelope";

export interface BrowserStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export function defaultBrowserStorage(): BrowserStorage {
  if (typeof window === "undefined") {
    throw new Error("browser storage is unavailable");
  }
  return window.localStorage;
}

export class RecoveryEnvelopeStore {
  constructor(private readonly storage: BrowserStorage) {}

  save(envelope: RecoveryEnvelope): RecoveryEnvelope {
    this.storage.setItem(RECOVERY_STORAGE_KEY, JSON.stringify(envelope));
    return structuredClone(envelope);
  }

  load(): RecoveryEnvelope | null {
    const encoded = this.storage.getItem(RECOVERY_STORAGE_KEY);
    if (encoded === null) return null;
    try {
      const parsed: unknown = JSON.parse(encoded);
      assertRecoveryEnvelope(parsed);
      return structuredClone(parsed);
    } catch {
      this.clear();
      throw new TypeError("The saved game was malformed and has been cleared.");
    }
  }

  clear(): void {
    this.storage.removeItem(RECOVERY_STORAGE_KEY);
  }
}
