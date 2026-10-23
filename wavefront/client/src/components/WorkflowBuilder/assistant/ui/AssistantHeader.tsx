import { Info, Sparkles, SquarePen, X } from 'lucide-react';
import React, { useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@app/components/ui/alert-dialog';
import { Button } from '@app/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@app/components/ui/select';
import type { WorkflowAssistant } from '../types';

interface AssistantHeaderProps {
  models: WorkflowAssistant['models'];
  selectedModelId?: string;
  onSelectModel: (id: string) => void;
  hasMessages: boolean;
  onNewChat: () => void;
  onClose: () => void;
}

export const AssistantHeader = ({
  models,
  selectedModelId,
  onSelectModel,
  hasMessages,
  onNewChat,
  onClose,
}: AssistantHeaderProps) => {
  const [confirming, setConfirming] = useState(false);

  return (
    <header className="border-border border-b px-3 py-2">
      <div className="flex items-center gap-2">
        <Sparkles size={14} className="text-muted-foreground shrink-0" aria-hidden />
        <h2 className="flex-1 truncate text-sm font-semibold">Workflow assistant</h2>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="h-7 gap-1 px-2 text-xs"
          disabled={!hasMessages}
          onClick={() => setConfirming(true)}
        >
          <SquarePen size={13} /> New chat
        </Button>
        <Button type="button" size="icon-xs" variant="ghost" onClick={onClose} aria-label="Close assistant">
          <X size={14} />
        </Button>
      </div>

      {models.length > 0 && (
        <div className="mt-2">
          <Select value={selectedModelId} onValueChange={onSelectModel}>
            <SelectTrigger aria-label="Model" className="h-8 text-xs">
              <SelectValue placeholder="Choose a model" />
            </SelectTrigger>
            <SelectContent>
              {models.map((model) => (
                <SelectItem key={model.id} value={model.id} className="text-xs">
                  {model.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-muted-foreground mt-1.5 flex items-start gap-1.5 text-[11px] leading-snug">
            <Info size={11} className="mt-0.5 shrink-0" aria-hidden />
            Models without reliable tool-calling may produce invalid workflows.
          </p>
        </div>
      )}

      <AlertDialog open={confirming} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Start a new chat?</AlertDialogTitle>
            <AlertDialogDescription>
              This clears the conversation and any proposals that are still pending. Changes you already applied stay on
              the canvas.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep chat</AlertDialogCancel>
            <AlertDialogAction onClick={onNewChat}>Start new chat</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </header>
  );
};
