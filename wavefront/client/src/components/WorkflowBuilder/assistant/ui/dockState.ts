export const DOCK_MIN_WIDTH = 320;
export const DOCK_MAX_WIDTH = 560;
export const DOCK_DEFAULT_WIDTH = 380;
// Below this viewport width the drawer overlays the canvas instead of taking space from it
export const DOCK_OVERLAY_BELOW = 1536;
export const DOCK_PREFS_KEY = 'workflow-assistant:dock';

export interface DockPrefs {
  open: boolean;
  width: number;
}

export const clampWidth = (width: number, viewportWidth?: number): number => {
  const max = viewportWidth ? Math.min(DOCK_MAX_WIDTH, Math.max(DOCK_MIN_WIDTH, viewportWidth - 48)) : DOCK_MAX_WIDTH;
  if (!Number.isFinite(width)) return DOCK_DEFAULT_WIDTH;
  return Math.round(Math.min(max, Math.max(DOCK_MIN_WIDTH, width)));
};

export const parseDockPrefs = (raw: string | null): DockPrefs => {
  const fallback: DockPrefs = { open: false, width: DOCK_DEFAULT_WIDTH };
  if (!raw) return fallback;
  try {
    const value = JSON.parse(raw) as Partial<DockPrefs> | null;
    return {
      open: value?.open === true,
      width: typeof value?.width === 'number' ? clampWidth(value.width) : DOCK_DEFAULT_WIDTH,
    };
  } catch {
    return fallback;
  }
};

export interface ShortcutEventLike {
  code: string;
  altKey: boolean;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
}

// Alt+Shift+A: unused by the browser and the builder (Ctrl/Cmd+L is the address bar)
export const ASSISTANT_SHORTCUT_LABEL = 'Alt+Shift+A';

export const isAssistantShortcut = (e: ShortcutEventLike): boolean =>
  e.code === 'KeyA' && e.altKey && e.shiftKey && !e.ctrlKey && !e.metaKey;

export interface TargetLike {
  tagName?: string;
  isContentEditable?: boolean;
  closest?: (selector: string) => unknown;
}

// The shortcut must not fire while the user types in a field (including the YAML code editor)
export const isTypingTarget = (target: TargetLike | null): boolean => {
  if (!target) return false;
  const tag = target.tagName?.toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  if (target.isContentEditable) return true;
  return Boolean(target.closest?.('.cm-editor'));
};
