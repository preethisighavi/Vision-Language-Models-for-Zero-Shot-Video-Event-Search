import { useEffect } from "react";

const TEAM = ["Abhinav Vummidichetty", "Ariel Hsieh", "Joseph Chang", "Preethi Sighavi", "Vaheedur Rehman Mahmed"];

export function AboutPage() {
  useEffect(() => {
    document.title = "About · Spartan Search";
  }, []);
  return (
    <div className="wrap-marketing pb-20 pt-12 sm:pt-16">
      <h1 className="h-page">About Spartan.</h1>
      <p className="lead mt-4 max-w-[720px]">
        Spartan Search is the demo for our capstone on zero-shot video and event search. You describe what happens in plain language; the system finds the
        video and the seconds that match, without any labels for your query.
      </p>

      <section className="card-lg mt-12 p-6 sm:p-8" aria-labelledby="arch">
        <h2 id="arch" className="h-card">How it works</h2>
        <p className="mt-2 max-w-[720px] text-text-secondary">
          Videos are split into short windows and encoded once. A query is encoded, compared with every window, re-ranked for the closest candidates, and then
          localized to the matching seconds.
        </p>
        <ArchitectureDiagram />
      </section>

      <div className="mt-6 grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
        <section className="card-lg p-6 sm:p-8" aria-labelledby="team">
          <h2 id="team" className="h-card">Team 1</h2>
          <p className="meta mt-1">DATA 298B · San José State University</p>
          <ul className="mt-4 flex flex-col gap-1.5">
            {TEAM.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </section>
        <section className="card-lg p-6 sm:p-8" aria-labelledby="credits">
          <h2 id="credits" className="h-card">Datasets and licences</h2>
          <ul className="mt-4 flex flex-col gap-3 text-[15px]">
            <li>
              <span className="font-semibold">MSR-VTT</span> (Xu et al., 2016). Research use only; videos remain the property of their owners.
            </li>
            <li>
              <span className="font-semibold">MSVD</span> (Chen and Dolan, 2011). Research use only.
            </li>
            <li>
              <span className="font-semibold">Uploads</span> are stored only for this demonstration and can be removed on request.
            </li>
          </ul>
          <p className="meta mt-4">This site is an academic demonstration and is not a commercial service.</p>
        </section>
      </div>
    </div>
  );
}

function ArchitectureDiagram() {
  const boxes = [
    { x: 10, label: "Query", sub: "plain sentence" },
    { x: 190, label: "Text encoder", sub: "one embedding" },
    { x: 370, label: "Stage 1", sub: "vector search" },
    { x: 550, label: "Re-rank", sub: "top candidates" },
    { x: 730, label: "Localize", sub: "start – end" },
  ];
  return (
    <svg viewBox="0 0 900 230" className="mt-6 w-full" role="img" aria-label="Pipeline: a query is encoded, searched against the video index, re-ranked, and localized to a time span. Videos are windowed and encoded offline into the index.">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="var(--text-secondary)" />
        </marker>
      </defs>
      {boxes.map((b, i) => (
        <g key={b.label}>
          <rect x={b.x} y={20} width={160} height={64} rx={16} fill={i === 4 ? "var(--accent-tint)" : "var(--bg)"} stroke="var(--hairline)" />
          <text x={b.x + 80} y={48} textAnchor="middle" fontSize={15} fontWeight={600} fill={i === 4 ? "var(--accent-text)" : "var(--text)"}>{b.label}</text>
          <text x={b.x + 80} y={68} textAnchor="middle" fontSize={12} fill="var(--text-secondary)">{b.sub}</text>
          {i < boxes.length - 1 && <line x1={b.x + 162} y1={52} x2={b.x + 186} y2={52} stroke="var(--text-secondary)" strokeWidth={1.5} markerEnd="url(#arrow)" />}
        </g>
      ))}
      <rect x={190} y={140} width={340} height={64} rx={16} fill="var(--surface-dark)" />
      <text x={360} y={168} textAnchor="middle" fontSize={15} fontWeight={600} fill="var(--text-on-dark)">Videos → windows → video encoder</text>
      <text x={360} y={188} textAnchor="middle" fontSize={12} fill="var(--text-on-dark-secondary)">offline, once per model</text>
      <rect x={570} y={140} width={180} height={64} rx={16} fill="var(--bg)" stroke="var(--hairline)" />
      <text x={660} y={168} textAnchor="middle" fontSize={15} fontWeight={600} fill="var(--text)">Vector index</text>
      <text x={660} y={188} textAnchor="middle" fontSize={12} fill="var(--text-secondary)">+ frame features</text>
      <line x1={532} y1={172} x2={566} y2={172} stroke="var(--text-secondary)" strokeWidth={1.5} markerEnd="url(#arrow)" />
      <path d="M660 138 V112 H450 V88" fill="none" stroke="var(--text-secondary)" strokeWidth={1.5} markerEnd="url(#arrow)" />
    </svg>
  );
}
