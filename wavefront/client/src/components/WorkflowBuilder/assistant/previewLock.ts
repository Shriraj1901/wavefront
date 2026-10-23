// While an assistant proposal is previewed, the live graph is read-only: the overlay was computed from it, so an edit
// underneath would leave the preview (and a later Apply) describing a workflow that no longer exists.
export const PREVIEW_LOCK_HINT = 'Close the preview to edit';

// The one check every graph edit goes through. Returns true when the edit must be dropped, after telling the user why
// through `notify` (the app's toast). Does nothing when no preview is open.
export const blockedByPreview = (previewing: boolean, notify: (message: string) => void): boolean => {
  if (!previewing) return false;
  notify(PREVIEW_LOCK_HINT);
  return true;
};
