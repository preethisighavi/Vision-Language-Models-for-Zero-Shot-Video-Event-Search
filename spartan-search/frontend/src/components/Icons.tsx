type P = { className?: string; size?: number };

const base = (size = 20) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
  focusable: false,
});

export const SearchIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="m20 20-4.2-4.2" />
  </svg>
);

export const PlayGlyph = ({ className, size = 18 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden focusable={false}>
    <path d="M8 5.5v13l10.5-6.5z" fill="currentColor" />
  </svg>
);

export const UploadIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M12 16V4" />
    <path d="m7 9 5-5 5 5" />
    <path d="M5 20h14" />
  </svg>
);

export const CheckIcon = ({ className, size = 14 }: P) => (
  <svg {...base(size)} strokeWidth={2.4} className={className}>
    <path d="m5 12.5 4.5 4.5L19 7.5" />
  </svg>
);

export const ChevronRight = ({ className, size = 16 }: P) => (
  <svg {...base(size)} className={className}>
    <path d="m9 6 6 6-6 6" />
  </svg>
);

export const AlertIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7.5v5.5" />
    <path d="M12 16.5h.01" />
  </svg>
);

export const ClockIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);
