import { Check, ChevronRight, CircleAlert, Loader2, X } from 'lucide-react';
import React, { useState } from 'react';
import type { TimelineStep } from '../types';
import { shouldExpandSteps, summarizeSteps } from './proposalState';

const StepIcon = ({ status, tone }: { status: TimelineStep['status']; tone?: TimelineStep['tone'] }) => {
  if (tone === 'warning' && status !== 'running')
    return <CircleAlert size={12} className="text-amber-600 dark:text-amber-400" aria-hidden />;
  if (status === 'running')
    return <Loader2 size={12} className="text-muted-foreground animate-spin motion-reduce:animate-none" aria-hidden />;
  if (status === 'failed') return <X size={12} className="text-destructive" aria-hidden />;
  return <Check size={12} className="text-muted-foreground" aria-hidden />;
};

const STATUS_TEXT = { running: 'Running', done: 'Done', failed: 'Failed' } as const;

const StepRow = ({ step }: { step: TimelineStep }) => {
  const [open, setOpen] = useState(false);
  return (
    <li className="text-xs">
      <div className="flex items-start gap-2">
        <span className="mt-0.5 shrink-0">
          <StepIcon status={step.status} tone={step.tone} />
        </span>
        <div className="min-w-0 flex-1">
          <span className={step.status === 'failed' ? 'text-destructive' : 'text-foreground'}>{step.label}</span>
          <span className="sr-only"> {STATUS_TEXT[step.status]}</span>
          {step.status === 'failed' && <span className="text-destructive ml-1.5 text-[11px]">Failed</span>}
          {step.tone === 'warning' && step.detail && (
            <span className="ml-1.5 text-[11px] text-amber-700 dark:text-amber-400">{step.detail}</span>
          )}
          {step.detail && step.tone !== 'warning' && (
            <button
              type="button"
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
              className="text-muted-foreground hover:text-foreground ml-2 text-[11px] underline-offset-2 hover:underline"
            >
              {open ? 'Hide details' : 'Details'}
            </button>
          )}
          {open && step.detail && step.tone !== 'warning' && (
            <pre className="bg-muted text-muted-foreground mt-1 max-h-32 overflow-auto rounded p-2 font-mono text-[11px] break-words whitespace-pre-wrap">
              {step.detail}
            </pre>
          )}
        </div>
      </div>
    </li>
  );
};

// Collapsible list of what the assistant did. Open while work is running or failed; collapsed once it settles
// cleanly. A manual toggle wins until the automatic state flips again.
export const StepTimeline = React.memo(({ steps }: { steps: TimelineStep[] }) => {
  const auto = shouldExpandSteps(steps);
  const [override, setOverride] = useState<{ auto: boolean; open: boolean } | null>(null);
  const open = override && override.auto === auto ? override.open : auto;
  if (steps.length === 0) return null;

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOverride({ auto, open: !open })}
        className="text-muted-foreground hover:text-foreground focus-visible:ring-ring/50 -ml-1 flex items-center gap-1 rounded px-1 py-0.5 text-xs outline-none focus-visible:ring-2"
      >
        <ChevronRight
          size={12}
          className={`transition-transform duration-150 motion-reduce:transition-none ${open ? 'rotate-90' : ''}`}
          aria-hidden
        />
        {summarizeSteps(steps)}
      </button>
      {open && (
        <ol className="border-border animate-in fade-in-0 mt-1 ml-1.5 space-y-1.5 border-l pl-3 duration-150 motion-reduce:animate-none">
          {steps.map((step) => (
            <StepRow key={step.id} step={step} />
          ))}
        </ol>
      )}
    </div>
  );
});
StepTimeline.displayName = 'StepTimeline';
