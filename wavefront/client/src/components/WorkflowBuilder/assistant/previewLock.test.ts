import { describe, expect, it, vi } from 'vitest';
import { blockedByPreview, PREVIEW_LOCK_HINT } from './previewLock';

describe('blockedByPreview', () => {
  it('lets an edit through without a message when nothing is previewed', () => {
    const notify = vi.fn();
    expect(blockedByPreview(false, notify)).toBe(false);
    expect(notify).not.toHaveBeenCalled();
  });

  it('drops the edit and says why while a preview is open', () => {
    const notify = vi.fn();
    expect(blockedByPreview(true, notify)).toBe(true);
    expect(notify).toHaveBeenCalledTimes(1);
    expect(notify).toHaveBeenCalledWith('Close the preview to edit');
    expect(PREVIEW_LOCK_HINT).toBe('Close the preview to edit');
  });
});
