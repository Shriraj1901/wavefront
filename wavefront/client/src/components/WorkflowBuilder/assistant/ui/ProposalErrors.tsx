import { CircleAlert } from 'lucide-react';
import React from 'react';
import type { ProposalError } from '../types';

// Same visual language as ProblemsPanel, sized for the drawer
export const ProposalErrors = React.memo(({ errors, title }: { errors: ProposalError[]; title?: string }) => {
  if (errors.length === 0) return null;
  return (
    <div role="group" aria-label="Proposal errors" className="border-destructive/50 bg-background rounded-md border">
      <div className="flex items-start gap-2 px-3 pt-2.5 pb-1.5">
        <CircleAlert size={14} className="text-destructive mt-0.5 shrink-0" aria-hidden />
        <p className="text-xs font-semibold">
          {title ?? `${errors.length} ${errors.length === 1 ? 'error' : 'errors'} in this proposal`}
        </p>
      </div>
      <ul className="max-h-40 space-y-1.5 overflow-y-auto px-3 pb-2.5 pl-8">
        {errors.map((error, index) => (
          <li key={`${error.path}-${index}`} className="text-xs leading-snug">
            <span className="break-words">{error.message}</span>
            {error.path && (
              <span className="text-muted-foreground mt-0.5 block font-mono text-[11px] break-all">{error.path}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
});
ProposalErrors.displayName = 'ProposalErrors';
