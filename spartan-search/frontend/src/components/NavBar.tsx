import { Link, NavLink } from "react-router-dom";

import { useTheme } from "../hooks/useTheme";

export function Logo() {
  return (
    <Link to="/" className="inline-flex items-center gap-2 rounded-md text-[17px] font-semibold tracking-[-0.02em]" aria-label="Spartan home">
      <span className="grid h-[22px] w-[22px] place-items-center rounded-[6px] bg-text" aria-hidden>
        <svg width="10" height="10" viewBox="0 0 10 10">
          <path d="M2.5 1.2v7.6L8.6 5z" fill="var(--surface)" />
        </svg>
      </span>
      Spartan
    </Link>
  );
}

const LINKS = [
  { to: "/search", label: "Search" },
  { to: "/compare", label: "Compare" },
  { to: "/upload", label: "Upload" },
  { to: "/library", label: "Library" },
  { to: "/benchmarks", label: "Benchmarks" },
];

export function NavBar() {
  const { theme, toggle } = useTheme();
  const dark = theme === "dark";
  return (
    <header
      className="sticky top-0 z-40 border-b border-hairline"
      style={{ background: "var(--nav-bg)", backdropFilter: "saturate(180%) blur(20px)", WebkitBackdropFilter: "saturate(180%) blur(20px)" }}
    >
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 focus:rounded-pill focus:bg-surface focus:px-4 focus:py-2 focus:shadow-md">
        Skip to content
      </a>
      <nav aria-label="Main" className="wrap-app flex min-h-[52px] flex-wrap items-center justify-between gap-x-6 gap-y-1 py-2 sm:py-0">
        <Logo />
        <ul className="flex flex-wrap items-center gap-x-5 gap-y-1 text-[13px] text-text-nav sm:gap-x-7">
          <li>
            <button
              type="button"
              onClick={toggle}
              aria-pressed={dark}
              aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
              className="grid h-8 w-8 place-items-center rounded-full text-text-nav transition-colors duration-150 ease-out hover:bg-fill hover:text-text"
            >
              {dark ? <SunIcon /> : <MoonIcon />}
            </button>
          </li>
          {LINKS.map((l) => (
            <li key={l.to}>
              <NavLink
                to={l.to}
                className={({ isActive }) =>
                  `inline-block rounded-md py-1 transition-colors duration-150 ease-out hover:text-text ${isActive ? "font-semibold text-text" : ""}`
                }
              >
                {l.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden focusable={false}>
      <path d="M20 14.5A8 8 0 1 1 9.5 4 6.5 6.5 0 0 0 20 14.5z" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden focusable={false}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 3v1.5M12 19.5V21M3 12h1.5M19.5 12H21M5.6 5.6l1 1M17.4 17.4l1 1M18.4 5.6l-1 1M6.6 17.4l-1 1" />
    </svg>
  );
}

export function Footer() {
  return (
    <footer className="mt-auto border-t border-hairline py-8 text-[13px] text-text-secondary">
      <div className="wrap-app flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <p>Spartan VLM · DATA 298B, Team 1 · San José State University</p>
        <p>
          Research datasets: MSR-VTT and MSVD. For academic demonstration.{" "}
          <Link to="/about" className="link">
            About
          </Link>
        </p>
      </div>
    </footer>
  );
}
