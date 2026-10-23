import React from 'react';
import { useNodeId } from '@xyflow/react';
import { Minus, Pencil, Plus, Trash2 } from 'lucide-react';
import type { NodeDiffStatus } from '../graphDiff';
import { useNodeDiffStatus } from '../assistant/useNodeDiffStatus';

interface NodeCardProps {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  colorClass: string;
  selected?: boolean;
  badge?: React.ReactNode;
  onDelete?: () => void;
}

// Assistant preview overlay: outline style differs per status (not color alone) and a text badge names it
const DIFF_STYLES: Record<NodeDiffStatus, { outline: string; label: string; icon: React.ElementType }> = {
  added: { outline: 'outline-2 outline-offset-2 outline-emerald-500', label: 'Added', icon: Plus },
  modified: { outline: 'outline-2 outline-offset-2 outline-dashed outline-amber-500', label: 'Modified', icon: Pencil },
  removed: {
    outline: 'outline-2 outline-offset-2 outline-dotted outline-red-500 opacity-70',
    label: 'Removed',
    icon: Minus,
  },
};

// Common card wrapper for all nodes
export const NodeCard = ({ title, icon: Icon, children, colorClass, selected, badge, onDelete }: NodeCardProps) => {
  const nodeId = useNodeId();
  const diffStatus = useNodeDiffStatus(nodeId ?? '');
  const diff = diffStatus ? DIFF_STYLES[diffStatus] : null;
  return (
    <div
      className={`bg-card relative w-72 rounded-xl border shadow-md transition-all ${
        selected ? 'border-primary ring-primary/20 shadow-lg ring-2' : 'border-border hover:border-border/80'
      } ${diff?.outline ?? ''}`}
    >
      {diff && (
        <span className="bg-background text-foreground border-border absolute -top-2.5 left-3 z-10 flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium shadow-xs">
          <diff.icon size={10} aria-hidden />
          {diff.label}
        </span>
      )}
      <div className={`border-border flex items-center justify-between rounded-t-xl border-b p-3 ${colorClass}`}>
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="bg-background/60 text-foreground flex h-6 w-6 shrink-0 items-center justify-center rounded shadow-xs">
            <Icon size={14} />
          </div>
          <h4 className="text-foreground truncate text-xs font-semibold">{title}</h4>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {badge}
          {onDelete && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              title="Delete node"
              className="text-muted-foreground hover:text-destructive hover:bg-background/50 rounded p-1 transition-colors"
            >
              <Trash2 size={13} />
            </button>
          )}
        </div>
      </div>

      <div className="bg-card/50 flex flex-col gap-2 rounded-b-xl p-3 text-xs">{children}</div>
    </div>
  );
};
