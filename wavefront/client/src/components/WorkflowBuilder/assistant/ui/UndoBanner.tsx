import { Undo2 } from 'lucide-react';
import { Button } from '@app/components/ui/button';

export const UndoBanner = ({ onUndo }: { onUndo: () => void }) => (
  <div
    role="status"
    className="border-border bg-muted/50 animate-in fade-in-0 flex items-center gap-2 border-b px-3 py-1.5 text-xs duration-150 motion-reduce:animate-none"
  >
    <Undo2 size={13} className="text-muted-foreground shrink-0" aria-hidden />
    <span className="flex-1">Assistant changes applied</span>
    <Button type="button" size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={onUndo}>
      Undo
    </Button>
  </div>
);
