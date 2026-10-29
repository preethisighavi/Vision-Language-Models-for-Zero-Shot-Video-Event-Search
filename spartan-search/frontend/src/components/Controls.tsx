import { forwardRef, useEffect, useId, useState } from "react";

import type { CollectionInfo, ModelInfo } from "../api/client";
import { SearchIcon } from "./Icons";

interface SearchBarProps {
  initial?: string;
  size?: "lg" | "md";
  onSubmit: (query: string) => void;
  label?: string;
  placeholder?: string;
  buttonLabel?: string;
}

export const SearchBar = forwardRef<HTMLInputElement, SearchBarProps>(function SearchBar(
  { initial = "", size = "md", onSubmit, label = "Describe what happens", placeholder = "Describe a moment, e.g. a man is cooking pasta", buttonLabel = "Search" },
  ref,
) {
  const [value, setValue] = useState(initial);
  const id = useId();
  useEffect(() => setValue(initial), [initial]);
  const lg = size === "lg";
  return (
    <form
      role="search"
      onSubmit={(e) => {
        e.preventDefault();
        const q = value.trim();
        if (q) onSubmit(q.slice(0, 200));
      }}
      className={`flex w-full items-center gap-2 rounded-pill bg-surface pl-5 pr-2 shadow-lg ring-1 ring-hairline-soft transition-shadow duration-200 ease-out focus-within:ring-2 focus-within:ring-accent ${lg ? "h-16" : "h-[52px]"}`}
    >
      <SearchIcon className="shrink-0 text-text-secondary" size={lg ? 22 : 20} />
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      <input
        ref={ref}
        id={id}
        type="search"
        value={value}
        maxLength={200}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder}
        autoComplete="off"
        enterKeyHint="search"
        className={`min-w-0 flex-1 bg-transparent placeholder:text-text-secondary focus:outline-none ${lg ? "text-[19px]" : "text-[17px]"}`}
        style={{ outline: "none" }}
      />
      <kbd className="hidden select-none rounded-md border border-hairline px-1.5 text-[12px] text-text-secondary md:inline" aria-hidden>
        /
      </kbd>
      <button type="submit" className={`btn-primary shrink-0 ${lg ? "h-12 px-6 text-[16px]" : "h-10"}`}>
        {buttonLabel}
      </button>
    </form>
  );
});

export function ModelSwitch({
  models,
  value,
  onChange,
  label = "Model",
}: {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
  label?: string;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex max-w-full flex-wrap rounded-pill bg-fill p-[3px]">
      {models.map((m) => {
        const selected = m.id === value;
        return (
          <button
            key={m.id}
            type="button"
            aria-pressed={selected}
            title={m.description}
            onClick={() => onChange(m.id)}
            className={`rounded-pill px-4 py-1.5 text-[13px] font-medium transition-all duration-200 ease-out ${
              selected ? "bg-surface text-text shadow-segment" : "text-text-nav hover:text-text"
            }`}
          >
            {m.name}
            {m.status !== "ready" && <span className="ml-1 text-text-secondary">({m.status})</span>}
          </button>
        );
      })}
    </div>
  );
}

export function LibrarySelect({
  collections,
  value,
  onChange,
}: {
  collections: CollectionInfo[];
  value: string;
  onChange: (id: string) => void;
}) {
  const id = useId();
  return (
    <div className="relative inline-flex items-center">
      <label htmlFor={id} className="sr-only">
        Library
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-[34px] cursor-pointer appearance-none rounded-pill bg-fill pl-4 pr-9 text-[13px] font-medium text-text transition-colors hover:bg-track"
      >
        {collections.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name} ({c.video_count.toLocaleString()})
          </option>
        ))}
      </select>
      <svg className="pointer-events-none absolute right-3.5 text-text-nav" width="10" height="6" viewBox="0 0 10 6" aria-hidden>
        <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      </svg>
    </div>
  );
}

export function RerankSwitch({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId();
  return (
    <div className="inline-flex items-center gap-2">
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-[26px] w-[44px] shrink-0 rounded-pill transition-colors duration-200 ease-out ${checked ? "bg-switch-on" : "bg-track"}`}
      >
        <span
          className={`absolute left-0 top-[2px] h-[22px] w-[22px] rounded-full bg-white shadow-segment transition-transform duration-200 ease-out ${checked ? "translate-x-[20px]" : "translate-x-[2px]"}`}
          aria-hidden
        />
      </button>
      <label htmlFor={id} className="cursor-pointer text-[13px] font-medium text-text-nav">
        Re-rank
      </label>
    </div>
  );
}
