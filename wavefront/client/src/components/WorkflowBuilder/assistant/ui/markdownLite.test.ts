import { describe, expect, it } from 'vitest';
import { completePartialMarkdown, parseInline, parseMarkdownLite, type Block } from './markdownLite';

describe('parseInline', () => {
  it('splits bold and code, leaves the rest as text', () => {
    expect(parseInline('a **b** `c` d')).toEqual([
      { kind: 'text', text: 'a ' },
      { kind: 'bold', text: 'b' },
      { kind: 'text', text: ' ' },
      { kind: 'code', text: 'c' },
      { kind: 'text', text: ' d' },
    ]);
  });
  it('keeps unmatched markers as text', () => {
    expect(parseInline('**open')).toEqual([{ kind: 'text', text: '**open' }]);
  });
});

describe('parseMarkdownLite', () => {
  it('parses paragraphs, headings, lists and code', () => {
    const blocks = parseMarkdownLite('# T\nline one\nline two\n\n- a\n- b\n\n1. x\n\n```yaml\nk: v\n```');
    expect(blocks.map((b) => b.kind)).toEqual(['heading', 'paragraph', 'list', 'list', 'code']);
    expect(blocks[2]).toMatchObject({ ordered: false });
    expect((blocks[2] as { items: unknown[] }).items).toHaveLength(2);
    expect(blocks[3]).toMatchObject({ ordered: true });
    expect(blocks[4]).toEqual({ kind: 'code', text: 'k: v' });
  });
  it('keeps the author numbering across interleaved bullets', () => {
    const blocks = parseMarkdownLite('1. a\n- x\n2. b\n- y\n3. c');
    const lists = blocks.filter((b) => b.kind === 'list');
    expect(lists.map((b) => (b as { ordered: boolean }).ordered)).toEqual([true, false, true, false, true]);
    expect(
      lists.filter((b) => (b as { ordered: boolean }).ordered).map((b) => (b as { start?: number }).start)
    ).toEqual([1, 2, 3]);
  });
  it('starts an ordered list at its first number and leaves bullets without one', () => {
    const [ordered, bullets] = parseMarkdownLite('4. d\n5. e\n\n- f');
    expect(ordered).toMatchObject({ ordered: true, start: 4 });
    expect((ordered as { items: unknown[] }).items).toHaveLength(2);
    expect(bullets).toEqual({ kind: 'list', ordered: false, items: [[{ kind: 'text', text: 'f' }]] });
  });
  it('treats an unclosed fence as code while streaming', () => {
    expect(parseMarkdownLite('```\nabc')).toEqual([{ kind: 'code', text: 'abc' }]);
  });
  it('never emits HTML: tags stay text', () => {
    const [block] = parseMarkdownLite('<script>alert(1)</script>');
    expect(block).toEqual({ kind: 'paragraph', inline: [{ kind: 'text', text: '<script>alert(1)</script>' }] });
  });
  it('handles empty input', () => {
    expect(parseMarkdownLite('')).toEqual([]);
  });
});

describe('completePartialMarkdown', () => {
  it.each([
    ['closes an open bold span', 'a **bol', 'a **bol**'],
    ['drops a lone bold marker', 'a **', 'a '],
    ['drops half a bold marker', 'a *', 'a '],
    ['treats a half closing marker as the open span it ends', '**bold*', '**bold**'],
    ['closes an open code span', 'x `co', 'x `co`'],
    ['drops a lone backtick', 'x `', 'x '],
    ['drops a partial fence marker', 'text\n``', 'text\n'],
    ['drops a partial closing fence inside an open fence', '```\nk: v\n``', '```\nk: v\n'],
    ['drops a bullet marker with nothing after it', '- a\n-', '- a\n'],
    ['drops a numbered marker with nothing after it', '1. a\n2.', '1. a\n'],
    ['drops a heading marker with nothing after it', '##', ''],
    ['closes bold inside a list item cut mid-word', '- **Tighten the en', '- **Tighten the en**'],
    ['leaves complete markdown alone', 'a **b** `c`\n- d', 'a **b** `c`\n- d'],
    ['leaves an open fence alone', '```yaml\nk: `v', '```yaml\nk: `v'],
    ['leaves earlier lines alone', '**open\nnext', '**open\nnext'],
    ['handles empty input', '', ''],
  ])('%s', (_name, input, expected) => {
    expect(completePartialMarkdown(input)).toBe(expected);
  });
});

describe('parseMarkdownLite with partial text', () => {
  it('renders an unclosed bold span as bold while it streams, and only then (not partial) as raw text', () => {
    expect(parseMarkdownLite('Use **bold te', { partial: true })).toEqual([
      {
        kind: 'paragraph',
        inline: [
          { kind: 'text', text: 'Use ' },
          { kind: 'bold', text: 'bold te' },
        ],
      },
    ]);
    expect(parseMarkdownLite('Use **bold te')).toEqual([
      { kind: 'paragraph', inline: [{ kind: 'text', text: 'Use **bold te' }] },
    ]);
  });

  it('renders an unclosed inline code span as code', () => {
    const [block] = parseMarkdownLite('Set `input_fil', { partial: true });
    expect(block).toEqual({
      kind: 'paragraph',
      inline: [
        { kind: 'text', text: 'Set ' },
        { kind: 'code', text: 'input_fil' },
      ],
    });
  });

  it('keeps an unclosed fence as code and hides a half-typed closing fence', () => {
    expect(parseMarkdownLite('```yaml\nk: v\n``', { partial: true })).toEqual([{ kind: 'code', text: 'k: v\n' }]);
    expect(parseMarkdownLite('Intro\n\n```', { partial: true }).map((b) => b.kind)).toEqual(['paragraph', 'code']);
  });

  it('shows a list item cut mid-word and does not add an empty item for a bare marker', () => {
    const [list] = parseMarkdownLite('- first\n- seco', { partial: true });
    expect(list).toMatchObject({ kind: 'list', ordered: false });
    expect((list as { items: unknown[][] }).items).toHaveLength(2);
    const [bare] = parseMarkdownLite('- first\n-', { partial: true });
    expect((bare as { items: unknown[][] }).items).toHaveLength(1);
  });

  it('keeps a trailing partial line as text', () => {
    const blocks = parseMarkdownLite('First line.\n\nSecond lin', { partial: true });
    expect(blocks).toHaveLength(2);
    expect(blocks[1]).toEqual({ kind: 'paragraph', inline: [{ kind: 'text', text: 'Second lin' }] });
  });

  it('never throws and never leaks raw markers for any prefix of a document', () => {
    const doc = [
      '## Plan',
      'I would change it in **three** steps, using `input_filter` too.',
      '- **Tighten the entry agent.** Its job is broad.',
      '- Add a `reviewer`.',
      '1. First',
      '2. Second',
      '```yaml\narium:\n  workflow:\n    start: entry\n```',
      'Done **now**.',
    ].join('\n\n');
    const inlineTexts = (b: Block): string[] => {
      if (b.kind === 'code') return [];
      const groups = b.kind === 'list' ? b.items : [b.inline];
      return groups.flatMap((tokens) => tokens.filter((t) => t.kind === 'text').map((t) => t.text));
    };
    for (let end = 0; end <= doc.length; end++) {
      const blocks = parseMarkdownLite(doc.slice(0, end), { partial: true });
      blocks.forEach((b) => {
        inlineTexts(b).forEach((text) => {
          expect(text).not.toContain('**');
          expect(text).not.toContain('`');
        });
        if (b.kind === 'list') b.items.forEach((item) => expect(item.length).toBeGreaterThan(0));
      });
    }
  });
});
