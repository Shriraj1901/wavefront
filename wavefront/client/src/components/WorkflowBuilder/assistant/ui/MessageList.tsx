import { ArrowDown, CircleAlert, RotateCcw } from 'lucide-react';
import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Button } from '@app/components/ui/button';
import type { AssistantMessage, AssistantStatus } from '../types';
import { MessageBubble, type ProposalActions } from './MessageBubble';
import { getTurnUserText } from './proposalState';
import { StarterPrompts } from './StarterPrompts';

interface MessageListProps {
  messages: AssistantMessage[];
  status: AssistantStatus;
  error?: string;
  actions: ProposalActions;
  onSend: (text: string) => void;
  onRetry: () => void;
}

const STICK_THRESHOLD = 48;
const CONTINUE_PROMPT = 'continue';

export const MessageList = ({ messages, status, error, actions, onSend, onRetry }: MessageListProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(true);
  const stuckRef = useRef(true);
  const streaming = status === 'streaming';

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_THRESHOLD;
    stuckRef.current = atBottom;
    setStuck((prev) => (prev === atBottom ? prev : atBottom));
  }, []);

  const jumpToLatest = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // No smooth scrolling while text is arriving: the target moves every frame and the animation would lag behind it
    el.scrollTo({
      top: el.scrollHeight,
      behavior: streaming || window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    });
  }, [streaming]);

  // Follow new content only while the user has not scrolled up
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el && stuckRef.current) el.scrollTop = el.scrollHeight;
  }, [messages, error, status]);

  // A new user message always brings the view back to the bottom
  const lastRole = messages[messages.length - 1]?.role;
  useEffect(() => {
    if (lastRole === 'user') {
      stuckRef.current = true;
      setStuck(true);
    }
  }, [lastRole, messages.length]);

  const lastMessage = messages[messages.length - 1];
  const lastId = lastMessage?.id;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        onScroll={onScroll}
        role="log"
        aria-label="Conversation"
        aria-live="polite"
        aria-busy={streaming}
        className="min-h-0 flex-1 overflow-y-auto"
      >
        {messages.length === 0 ? (
          <StarterPrompts onPick={onSend} />
        ) : (
          <div className="space-y-4 px-3 py-3">
            {messages.map((message, index) => (
              <MessageBubble
                key={message.id}
                message={message}
                redoText={message.proposal ? getTurnUserText(messages, index) : undefined}
                busy={streaming && !!message.proposal && message.id === lastId}
                streaming={streaming && message.id === lastId}
                actions={actions}
              />
            ))}
            {status === 'idle' && lastMessage?.role === 'assistant' && lastMessage.truncated && (
              <div className="flex items-center gap-2">
                <p className="text-muted-foreground text-xs">This reply was cut off at the output limit.</p>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-6 px-2 text-xs"
                  onClick={() => onSend(CONTINUE_PROMPT)}
                >
                  Continue
                </Button>
              </div>
            )}
            {streaming && messages[messages.length - 1]?.role === 'user' && (
              <p className="text-muted-foreground text-xs">Thinking...</p>
            )}
            {error && (
              <div
                role="alert"
                className="border-destructive/50 bg-background flex items-start gap-2 rounded-md border px-3 py-2"
              >
                <CircleAlert size={14} className="text-destructive mt-0.5 shrink-0" aria-hidden />
                <p className="min-w-0 flex-1 text-xs leading-snug break-words">{error}</p>
                {status === 'error' && (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-6 gap-1 px-2 text-xs"
                    onClick={onRetry}
                  >
                    <RotateCcw size={11} /> Retry
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
      </div>
      {/* A row of its own above the composer, so the button never covers message text */}
      {!stuck && messages.length > 0 && (
        <div className="border-border flex shrink-0 justify-center border-t py-1.5">
          <Button
            type="button"
            size="sm"
            variant="secondary"
            onClick={jumpToLatest}
            className="animate-in fade-in-0 h-7 gap-1 text-xs duration-150 motion-reduce:animate-none"
          >
            <ArrowDown size={12} /> Jump to latest
          </Button>
        </div>
      )}
    </div>
  );
};
