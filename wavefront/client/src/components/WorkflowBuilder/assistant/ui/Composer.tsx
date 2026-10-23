import { ArrowUp, Square } from 'lucide-react';
import React, { useCallback, useLayoutEffect, useState } from 'react';
import { Button } from '@app/components/ui/button';
import { Textarea } from '@app/components/ui/textarea';

interface ComposerProps {
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}

const MAX_HEIGHT = 160;

export const Composer = ({ streaming, onSend, onStop, inputRef }: ComposerProps) => {
  const [text, setText] = useState('');
  const canSend = text.trim().length > 0 && !streaming;

  // Grow with the content up to a cap, then scroll
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [text, inputRef]);

  const submit = useCallback(() => {
    if (!canSend) return;
    onSend(text);
    setText('');
  }, [canSend, onSend, text]);

  return (
    <div className="border-border border-t p-3">
      <div className="flex items-end gap-2">
        <Textarea
          ref={inputRef}
          value={text}
          rows={1}
          aria-label="Message the assistant"
          placeholder={streaming ? 'Waiting for the reply...' : 'Describe a change to your workflow'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          className="max-h-40 min-h-9 resize-none py-2 leading-snug"
        />
        {streaming ? (
          <Button type="button" size="icon-sm" variant="outline" onClick={onStop} aria-label="Stop" title="Stop">
            <Square size={13} fill="currentColor" />
          </Button>
        ) : (
          <Button type="button" size="icon-sm" disabled={!canSend} onClick={submit} aria-label="Send" title="Send">
            <ArrowUp size={15} />
          </Button>
        )}
      </div>
      <p className="text-muted-foreground mt-1.5 text-[11px]">Enter to send, Shift+Enter for a new line</p>
    </div>
  );
};
