// Minimal, safe markdown for assistant text: no HTML, no links, no dependency (the client has no markdown renderer).
// Supports paragraphs, #-headings, bullet/numbered lists, fenced code, **bold** and `inline code`.
// Text that is still streaming can end mid-syntax: pass { partial: true } to render that edge gracefully.

export type InlineToken = { kind: 'text' | 'bold' | 'code'; text: string };

export type Block =
  | { kind: 'paragraph'; inline: InlineToken[] }
  | { kind: 'heading'; inline: InlineToken[] }
  // start: the author's number for the first item of an ordered list, so numbering survives interleaved bullets
  | { kind: 'list'; ordered: boolean; items: InlineToken[][]; start?: number }
  | { kind: 'code'; text: string };

export const parseInline = (text: string): InlineToken[] =>
  text
    .split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/)
    .filter((part) => part.length > 0)
    .map((part): InlineToken => {
      if (part.length > 2 && part.startsWith('`') && part.endsWith('`'))
        return { kind: 'code', text: part.slice(1, -1) };
      if (part.length > 4 && part.startsWith('**') && part.endsWith('**'))
        return { kind: 'bold', text: part.slice(2, -2) };
      return { kind: 'text', text: part };
    });

const BULLET = /^\s*[-*]\s+(.*)$/;
const ORDERED = /^\s*(\d+)[.)]\s+(.*)$/;

const LONE_MARKER = /^\s*([-*]|#{1,6}|\d+[.)])\s*$/;

// Makes the end of half-written markdown renderable, so streamed text does not flicker as markers arrive: a bold or
// code span that is still open is closed, a marker typed so far (`**`, a lone backtick, `-`, `1.`, `##`) is dropped,
// and a partial fence marker (` or ``) is dropped. Only the last line is touched. An open fence is left alone: the
// parser already runs it to the end.
export const completePartialMarkdown = (source: string): string => {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const index = lines.length - 1;
  let line = lines[index];

  if (/^`{1,2}$/.test(line.trim())) {
    lines[index] = '';
    return lines.join('\n');
  }

  const inFence = lines.reduce((open, l) => (l.trimStart().startsWith('```') ? !open : open), false);
  if (inFence) return lines.join('\n');

  // A single trailing `*` may be half of a `**`
  if (/[^*]\*$/.test(line)) line = line.slice(0, -1);
  if ((line.match(/`/g) ?? []).length % 2 === 1) line = line.endsWith('`') ? line.slice(0, -1) : `${line}\``;
  const bold = (line.replace(/`[^`\n]*`/g, '').match(/\*\*/g) ?? []).length;
  if (bold % 2 === 1) line = line.endsWith('**') ? line.slice(0, -2) : `${line}**`;
  if (LONE_MARKER.test(line)) line = '';

  lines[index] = line;
  return lines.join('\n');
};

export const parseMarkdownLite = (source: string, options: { partial?: boolean } = {}): Block[] => {
  const blocks: Block[] = [];
  const text = options.partial ? completePartialMarkdown(source) : source;
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: InlineToken[][]; start?: number } | null = null;

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ kind: 'paragraph', inline: parseInline(paragraph.join(' ')) });
    paragraph = [];
  };
  const flushList = () => {
    if (list) blocks.push({ kind: 'list', ...list });
    list = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trimStart().startsWith('```')) {
      flushParagraph();
      flushList();
      const code: string[] = [];
      // An unclosed fence (still streaming) runs to the end
      for (i++; i < lines.length && !lines[i].trimStart().startsWith('```'); i++) code.push(lines[i]);
      blocks.push({ kind: 'code', text: code.join('\n') });
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    const heading = /^#{1,6}\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: 'heading', inline: parseInline(heading[1]) });
      continue;
    }
    const bullet = BULLET.exec(line);
    const ordered = bullet ? null : ORDERED.exec(line);
    if (bullet || ordered) {
      flushParagraph();
      const isOrdered = !bullet;
      if (list && list.ordered !== isOrdered) flushList();
      if (!list) list = { ordered: isOrdered, items: [], ...(isOrdered ? { start: Number(ordered?.[1]) } : {}) };
      list.items.push(parseInline(bullet ? bullet[1] : (ordered?.[2] ?? '')));
      continue;
    }
    flushList();
    paragraph.push(line.trim());
  }
  flushParagraph();
  flushList();
  return blocks;
};
