import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import {
  clampWidth,
  DOCK_PREFS_KEY,
  isAssistantShortcut,
  isTypingTarget,
  parseDockPrefs,
  type DockPrefs,
} from './dockState';

interface AssistantDock {
  open: boolean;
  width: number;
  setOpen: (open: boolean) => void;
  toggle: () => void;
  setWidth: (width: number) => void;
}

const noop = () => {};
const NO_DOCK: AssistantDock = { open: false, width: 0, setOpen: noop, toggle: noop, setWidth: noop };

const AssistantDockContext = createContext<AssistantDock>(NO_DOCK);

export const useAssistantDock = () => useContext(AssistantDockContext);

const readPrefs = (): DockPrefs => {
  try {
    return parseDockPrefs(localStorage.getItem(DOCK_PREFS_KEY));
  } catch {
    return parseDockPrefs(null);
  }
};

// Open/closed and width of the assistant drawer (per-viewer, localStorage only) plus the global toggle shortcut.
// Shared by the canvas toolbar button and the drawer, which live in different subtrees.
export const AssistantDockProvider = ({ children }: { children: React.ReactNode }) => {
  const [prefs, setPrefs] = useState<DockPrefs>(readPrefs);

  useEffect(() => {
    try {
      localStorage.setItem(DOCK_PREFS_KEY, JSON.stringify(prefs));
    } catch {
      // Not persisted: fine
    }
  }, [prefs]);

  const setOpen = useCallback((open: boolean) => setPrefs((p) => (p.open === open ? p : { ...p, open })), []);
  const toggle = useCallback(() => setPrefs((p) => ({ ...p, open: !p.open })), []);
  const setWidth = useCallback(
    (width: number) => setPrefs((p) => ({ ...p, width: clampWidth(width, window.innerWidth) })),
    []
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!isAssistantShortcut(e) || isTypingTarget(e.target as HTMLElement | null)) return;
      e.preventDefault();
      toggle();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [toggle]);

  const value = useMemo<AssistantDock>(
    () => ({ open: prefs.open, width: prefs.width, setOpen, toggle, setWidth }),
    [prefs.open, prefs.width, setOpen, toggle, setWidth]
  );

  return <AssistantDockContext.Provider value={value}>{children}</AssistantDockContext.Provider>;
};
