const STARTERS = [
  'Audit this workflow',
  'Add a fallback router',
  'Add an agent that summarizes the final output',
  'Explain what this workflow does',
];

export const StarterPrompts = ({ onPick }: { onPick: (text: string) => void }) => (
  <div className="flex flex-col gap-3 px-4 py-6">
    <div>
      <h3 className="text-sm font-semibold">Describe a workflow or an edit</h3>
      <p className="text-muted-foreground mt-1 text-xs leading-relaxed">
        The assistant reads the workflow on your canvas and proposes changes. Nothing changes until you apply them.
      </p>
    </div>
    <ul className="flex flex-col gap-1.5">
      {STARTERS.map((text) => (
        <li key={text}>
          <button
            type="button"
            onClick={() => onPick(text)}
            className="border-border hover:bg-muted focus-visible:ring-ring/50 w-full rounded-md border px-3 py-2 text-left text-xs outline-none focus-visible:ring-2"
          >
            {text}
          </button>
        </li>
      ))}
    </ul>
  </div>
);
