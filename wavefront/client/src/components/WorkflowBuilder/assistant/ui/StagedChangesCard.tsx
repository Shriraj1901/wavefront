import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  FileDiff,
  Minus,
  Pencil,
  Plus,
  X,
} from 'lucide-react';
import React, { useState } from 'react';
import { Button } from '@app/components/ui/button';
import type { Proposal } from '../types';
import { ProposalErrors } from './ProposalErrors';
import {
  REDO_LABEL,
  canPreview,
  canRedo,
  getApplyState,
  getCardState,
  getDiffTotals,
  getPreviewDisabledReason,
  getProposalNotes,
  getRemainingIssues,
  groupDiffEntries,
  listDiffNodes,
  summarizeDiffText,
  type DiffNodeEntry,
} from './proposalState';

export interface ProposalCardProps {
  proposal: Proposal;
  // The assistant is mid-reply
  busy: boolean;
  previewing: boolean;
  onTogglePreview: (id: string) => void;
  onApply: (id: string) => void;
  onDiscard: (id: string) => void;
  onFocusNode: (nodeId: string) => void;
  // Sends a follow-up message ("Help me fix these"). Optional: without it the button is not shown.
  onSend?: (text: string) => void;
  // What the user asked in the turn that made this proposal: sent again by "Redo on the current workflow" when the
  // proposal is stale. Without it (or onSend) no redo is offered.
  redoText?: string;
}

const COLLAPSED_NODES = 8;
export const FIX_ISSUES_PROMPT = 'Fix the remaining validation issues in this workflow.';

const STATUS_META: Record<DiffNodeEntry['status'], { label: string; icon: typeof Plus }> = {
  added: { label: 'Added', icon: Plus },
  modified: { label: 'Modified', icon: Pencil },
  removed: { label: 'Removed', icon: Minus },
};

const Count = ({ n, label }: { n: number; label: string }) =>
  n > 0 ? (
    <span className="bg-muted text-foreground rounded px-1.5 py-0.5 text-[11px] tabular-nums">
      {n} {label}
    </span>
  ) : null;

const SectionHeading = ({ children, hint }: { children: React.ReactNode; hint?: string }) => (
  <div className="px-3 pt-2.5">
    <h5 className="text-[11px] font-semibold tracking-wide uppercase">{children}</h5>
    {hint && <p className="text-muted-foreground text-[11px] leading-snug">{hint}</p>}
  </div>
);

const NodeRow = ({ entry, onFocusNode }: { entry: DiffNodeEntry; onFocusNode: (nodeId: string) => void }) => {
  const { label, icon: Icon } = STATUS_META[entry.status];
  const { name, changed, lines } = entry;
  return (
    <li>
      <button
        type="button"
        onClick={() => onFocusNode(name)}
        title={`Show ${name} on the canvas`}
        className="hover:bg-muted focus-visible:ring-ring/50 flex w-full items-start gap-2 px-2 py-1.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-inset"
      >
        <Icon size={12} className="text-muted-foreground mt-0.5 shrink-0" aria-hidden />
        <span className="min-w-0 flex-1">
          <span className="block truncate font-mono">{name}</span>
          {lines && lines.length > 0 ? (
            <span className="mt-0.5 block space-y-0.5">
              {lines.map((line) => (
                <span key={line.field} className="block text-[11px] leading-snug break-words">
                  <span className="font-medium">{line.field}:</span>{' '}
                  {line.before === line.after ? (
                    <span className="text-muted-foreground">edited (long text)</span>
                  ) : (
                    <>
                      <span className="text-muted-foreground">{line.before}</span>{' '}
                      <ArrowRight size={10} className="inline align-[-1px]" aria-label="changes to" />{' '}
                      <span>{line.after}</span>
                    </>
                  )}
                </span>
              ))}
            </span>
          ) : (
            changed && <span className="text-muted-foreground block truncate text-[11px]">Changed: {changed}</span>
          )}
        </span>
        <span className="text-muted-foreground shrink-0 text-[11px]">{label}</span>
      </button>
    </li>
  );
};

const NodeList = ({ entries, onFocusNode }: { entries: DiffNodeEntry[]; onFocusNode: (nodeId: string) => void }) =>
  entries.length > 0 ? (
    <ul className="mx-3 mt-1.5 divide-y rounded border text-xs">
      {entries.map((entry) => (
        <NodeRow key={`${entry.status}-${entry.name}`} entry={entry} onFocusNode={onFocusNode} />
      ))}
    </ul>
  ) : null;

export const StagedChangesCard = React.memo(
  ({
    proposal,
    busy,
    previewing,
    onTogglePreview,
    onApply,
    onDiscard,
    onFocusNode,
    onSend,
    redoText,
  }: ProposalCardProps) => {
    const [showAll, setShowAll] = useState(false);
    const [showExisting, setShowExisting] = useState(false);
    const cardState = getCardState(proposal);
    const totals = getDiffTotals(proposal.diff);

    if (cardState === 'applied' || cardState === 'discarded') {
      const applied = cardState === 'applied';
      const Icon = applied ? Check : X;
      return (
        <div
          className="border-border bg-card text-muted-foreground flex items-center gap-2 rounded-md border px-3 py-2 text-xs"
          role="status"
        >
          <Icon size={13} className="shrink-0" aria-hidden />
          <span className="text-foreground font-medium">{applied ? 'Applied' : 'Discarded'}</span>
          <span className="truncate">{summarizeDiffText(proposal.diff)}</span>
        </div>
      );
    }

    const apply = getApplyState(proposal, busy ? 'streaming' : 'idle');
    const { items, hidden } = listDiffNodes(proposal.diff, showAll ? Number.MAX_SAFE_INTEGER : COLLAPSED_NODES);
    const groups = groupDiffEntries(items);
    const notes = getProposalNotes(proposal);
    const reasonId = `apply-reason-${proposal.id}`;
    const previewable = canPreview(proposal);
    // Validation problems left in the proposed workflow, introduced or already there: Apply stays available
    const remaining = getRemainingIssues(proposal);
    const remainingCount = `${remaining.length} validation ${remaining.length === 1 ? 'issue' : 'issues'}`;
    const lead =
      cardState === 'pending-invalid'
        ? `I made the change but ${remainingCount} ${remaining.length === 1 ? 'remains' : 'remain'}.`
        : `This workflow still has ${remainingCount}.`;
    const one = remaining.length === 1;
    const remainingText = `${lead} The workflow will not save until ${one ? 'it is' : 'they are'} fixed. You can still apply it and fix ${one ? 'it' : 'them'} in the editor.`;

    return (
      <section aria-label="Proposed changes" className="border-border bg-card rounded-md border">
        <div className="flex items-center gap-2 px-3 pt-2.5">
          <FileDiff size={14} className="text-muted-foreground shrink-0" aria-hidden />
          <h4 className="text-xs font-semibold">Proposed changes</h4>
        </div>
        <p className="text-muted-foreground px-3 pt-1 text-xs">
          {previewing
            ? 'Previewing on the canvas: new steps show as ghosts and changed steps show their proposed values. Nothing changes until you press Apply.'
            : 'This is a preview. Nothing changes on the canvas until you press Apply.'}
        </p>

        <div className="flex flex-wrap gap-1.5 px-3 pt-2">
          {totals.isEmpty ? (
            <span className="text-muted-foreground text-xs">No changes to the canvas.</span>
          ) : (
            <>
              <Count n={totals.added} label="added" />
              <Count n={totals.modified} label="modified" />
              <Count n={totals.removed} label="removed" />
              <Count n={totals.edgesAdded} label={totals.edgesAdded === 1 ? 'edge added' : 'edges added'} />
              <Count n={totals.edgesRemoved} label={totals.edgesRemoved === 1 ? 'edge removed' : 'edges removed'} />
            </>
          )}
        </div>

        {/* No inner scroller: the card grows and the message list scrolls, so no row is ever clipped */}
        {groups.requested.length > 0 || totals.edgesAdded > 0 ? (
          <>
            <SectionHeading hint="New steps and connections for what you asked.">Requested</SectionHeading>
            <NodeList entries={groups.requested} onFocusNode={onFocusNode} />
            {totals.edgesAdded > 0 && (
              <p className="text-muted-foreground px-3 pt-1.5 text-[11px]">
                {totals.edgesAdded} new {totals.edgesAdded === 1 ? 'connection' : 'connections'}
              </p>
            )}
          </>
        ) : null}
        {groups.alsoChanged.length > 0 && (
          <>
            <SectionHeading hint="Existing steps this change touches. Check them before you apply.">
              {groups.requested.length > 0 ? 'Also changed' : 'Changed'}
            </SectionHeading>
            <NodeList entries={groups.alsoChanged} onFocusNode={onFocusNode} />
          </>
        )}
        {groups.removed.length > 0 && (
          <>
            <SectionHeading>Removed</SectionHeading>
            <NodeList entries={groups.removed} onFocusNode={onFocusNode} />
          </>
        )}
        {hidden > 0 && !showAll && (
          <button
            type="button"
            onClick={() => setShowAll(true)}
            className="text-muted-foreground hover:text-foreground mx-3 mt-1 text-xs underline-offset-2 hover:underline"
          >
            +{hidden} more
          </button>
        )}

        {notes.length > 0 && (
          <>
            <SectionHeading hint="You can change any of these in the node settings, or ask me to adjust.">
              Choices I made
            </SectionHeading>
            <ul className="mx-3 mt-1.5 list-disc space-y-1 pl-5 text-xs leading-snug">
              {notes.map((note, i) => (
                <li key={`${i}-${note}`} className="break-words">
                  {note}
                </li>
              ))}
            </ul>
          </>
        )}

        {remaining.length > 0 && (
          <div
            role="group"
            aria-label="Remaining validation issues"
            className="mx-3 mt-2 rounded-md border border-amber-500/50"
          >
            <div className="flex items-start gap-2 px-3 pt-2.5 pb-2">
              <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-500" aria-hidden />
              <p className="text-xs leading-snug">{remainingText}</p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5 px-3 pb-2.5 pl-8">
              <button
                type="button"
                onClick={() => setShowExisting((v) => !v)}
                aria-expanded={showExisting}
                className="text-muted-foreground hover:text-foreground flex items-center gap-0.5 text-xs underline-offset-2 hover:underline"
              >
                {showExisting ? <ChevronDown size={12} aria-hidden /> : <ChevronRight size={12} aria-hidden />}
                {showExisting ? 'Hide issues' : 'Show issues'}
              </button>
              {onSend && (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-6 text-xs"
                  disabled={busy}
                  onClick={() => onSend(FIX_ISSUES_PROMPT)}
                >
                  Help me fix these
                </Button>
              )}
            </div>
            {showExisting && (
              <div className="px-3 pb-2.5">
                <ProposalErrors errors={remaining} title="Validation issues" />
              </div>
            )}
          </div>
        )}

        {/* Stays at the bottom of the visible message list while the card scrolls past, so Apply is never hidden */}
        <div className="bg-card border-border sticky bottom-0 z-10 mt-2.5 rounded-b-md border-t">
          <div className="flex flex-wrap items-center gap-1.5 px-3 pt-2.5 pb-2.5">
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 gap-1 text-xs"
              aria-pressed={previewing}
              disabled={!previewable}
              title={getPreviewDisabledReason(proposal)}
              onClick={() => onTogglePreview(proposal.id)}
            >
              {previewing ? <EyeOff size={13} /> : <Eye size={13} />}
              {previewing ? 'Hide preview' : 'Preview'}
            </Button>
            <Button
              type="button"
              size="sm"
              className="h-7 text-xs"
              disabled={!apply.enabled}
              aria-describedby={apply.reason ? reasonId : undefined}
              onClick={() => onApply(proposal.id)}
            >
              Apply
            </Button>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-7 text-xs"
              onClick={() => onDiscard(proposal.id)}
            >
              Discard
            </Button>
            {onSend && redoText && canRedo(proposal, redoText) && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-7 text-xs"
                disabled={busy}
                onClick={() => onSend(redoText)}
              >
                {REDO_LABEL}
              </Button>
            )}
          </div>
          {apply.reason && (
            <p id={reasonId} className="text-muted-foreground px-3 pb-2.5 text-xs">
              {apply.reason}
            </p>
          )}
        </div>
      </section>
    );
  }
);
StagedChangesCard.displayName = 'StagedChangesCard';

// Extension point: other proposal kinds (for example a resource-approval card) register here. MessageBubble looks a
// card up by kind and renders nothing for kinds it does not know.
export const proposalCards: Record<string, React.ComponentType<ProposalCardProps>> = {
  staged_changes: StagedChangesCard,
};
