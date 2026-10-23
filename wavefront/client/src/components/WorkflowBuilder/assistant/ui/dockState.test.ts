import { describe, expect, it } from 'vitest';
import {
  clampWidth,
  DOCK_DEFAULT_WIDTH,
  DOCK_MAX_WIDTH,
  DOCK_MIN_WIDTH,
  isAssistantShortcut,
  isTypingTarget,
  parseDockPrefs,
} from './dockState';

describe('clampWidth', () => {
  it('clamps to the range and the viewport', () => {
    expect(clampWidth(100)).toBe(DOCK_MIN_WIDTH);
    expect(clampWidth(9999)).toBe(DOCK_MAX_WIDTH);
    expect(clampWidth(400)).toBe(400);
    expect(clampWidth(500, 500)).toBe(452);
    expect(clampWidth(NaN)).toBe(DOCK_DEFAULT_WIDTH);
  });
});

describe('parseDockPrefs', () => {
  it('falls back on bad input', () => {
    expect(parseDockPrefs(null)).toEqual({ open: false, width: DOCK_DEFAULT_WIDTH });
    expect(parseDockPrefs('{nope')).toEqual({ open: false, width: DOCK_DEFAULT_WIDTH });
  });
  it('reads and clamps stored prefs', () => {
    expect(parseDockPrefs('{"open":true,"width":9000}')).toEqual({ open: true, width: DOCK_MAX_WIDTH });
    expect(parseDockPrefs('{"open":"yes"}').open).toBe(false);
  });
});

describe('shortcut', () => {
  const ev = { code: 'KeyA', altKey: true, shiftKey: true, ctrlKey: false, metaKey: false };
  it('matches only Alt+Shift+A', () => {
    expect(isAssistantShortcut(ev)).toBe(true);
    expect(isAssistantShortcut({ ...ev, ctrlKey: true })).toBe(false);
    expect(isAssistantShortcut({ ...ev, code: 'KeyB' })).toBe(false);
  });
  it('ignores typing targets', () => {
    expect(isTypingTarget({ tagName: 'TEXTAREA' })).toBe(true);
    expect(isTypingTarget({ tagName: 'DIV', isContentEditable: true })).toBe(true);
    expect(isTypingTarget({ tagName: 'DIV', closest: () => ({}) })).toBe(true);
    expect(isTypingTarget({ tagName: 'BUTTON', closest: () => null })).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });
});
