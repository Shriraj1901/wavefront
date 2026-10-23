import React from 'react';
import type { AssistantMessage } from '../types';
import { parseMarkdownLite, type InlineToken } from './markdownLite';
import { proposalCards } from './StagedChangesCard';
import { StepTimeline } from './StepTimeline';

export interface ProposalActions {
  previewId: string | null;
  onTogglePreview: (id: string) => void;
  onApply: (id: string) => void;
  onDiscard: (id: string) => void;
  onFocusNode: (nodeId: string) => void;
  // Sends a follow-up message (the card's "Help me fix these")
  onSend: (text: string) => void;
}

interface MessageBubbleProps {
  message: AssistantMessage;
  // Only the proposal card cares whether a reply is running, so other messages always get false
  busy: boolean;
  // Additive: this message's text is still arriving, so render its unfinished markdown edge gracefully
  streaming?: boolean;
  actions: ProposalActions;
  // Additive: what the user asked in this reply's turn, for the proposal card's "Redo on the current workflow"
  redoText?: string;
}

const Inline = ({ tokens }: { tokens: InlineToken[] }) => (
  <>
    {tokens.map((t, i) =>
      t.kind === 'bold' ? (
        <strong key={i}>{t.text}</strong>
      ) : t.kind === 'code' ? (
        <code key={i} className="bg-muted rounded px-1 py-0.5 font-mono text-[0.85em]">
          {t.text}
        </code>
      ) : (
        <React.Fragment key={i}>{t.text}</React.Fragment>
      )
    )}
  </>
);

// partial: the text is still streaming and may end mid-syntax
const RichText = React.memo(({ text, partial }: { text: string; partial: boolean }) => {
  const blocks = parseMarkdownLite(text, { partial });
  return (
    <div className="space-y-2 text-sm leading-relaxed break-words">
      {blocks.map((b, i) => {
        switch (b.kind) {
          case 'heading':
            return (
              <p key={i} className="font-semibold">
                <Inline tokens={b.inline} />
              </p>
            );
          case 'list': {
            const items = b.items.map((item, j) => (
              <li key={j}>
                <Inline tokens={item} />
              </li>
            ));
            return b.ordered ? (
              <ol key={i} className="list-decimal space-y-0.5 pl-5" start={b.start}>
                {items}
              </ol>
            ) : (
              <ul key={i} className="list-disc space-y-0.5 pl-5">
                {items}
              </ul>
            );
          }
          case 'code':
            return (
              <pre
                key={i}
                className="bg-muted max-h-64 overflow-auto rounded-md p-2.5 font-mono text-xs whitespace-pre"
              >
                {b.text}
              </pre>
            );
          default:
            return (
              <p key={i}>
                <Inline tokens={b.inline} />
              </p>
            );
        }
      })}
    </div>
  );
});
RichText.displayName = 'RichText';

// Memoized by message identity: the reducer replaces only the streaming message, so settled bubbles never re-render
export const MessageBubble = React.memo(
  ({ message, busy, streaming = false, actions, redoText }: MessageBubbleProps) => {
    if (message.role === 'user') {
      return (
        <div className="flex justify-end">
          <div className="bg-primary/10 text-foreground max-w-[85%] rounded-lg rounded-br-sm px-3 py-2 text-sm leading-relaxed break-words whitespace-pre-wrap">
            {message.text}
          </div>
        </div>
      );
    }

    const { proposal } = message;
    // Unknown proposal kinds have no card and are ignored
    const Card = proposal ? proposalCards.staged_changes : undefined;

    return (
      <div className="space-y-2">
        <StepTimeline steps={message.steps} />
        {message.text && <RichText text={message.text} partial={streaming} />}
        {proposal && Card && (
          <Card
            proposal={proposal}
            busy={busy}
            previewing={actions.previewId === proposal.id}
            onTogglePreview={actions.onTogglePreview}
            onApply={actions.onApply}
            onDiscard={actions.onDiscard}
            onFocusNode={actions.onFocusNode}
            onSend={actions.onSend}
            redoText={redoText}
          />
        )}
      </div>
    );
  }
);
MessageBubble.displayName = 'MessageBubble';
