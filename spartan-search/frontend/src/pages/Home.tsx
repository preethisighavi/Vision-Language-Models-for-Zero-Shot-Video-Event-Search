import { useRef } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { LibrarySelect, ModelSwitch, SearchBar } from "../components/Controls";
import { ChevronRight } from "../components/Icons";
import { useCollections, useModels, useSlashFocus } from "../hooks/queries";
import { resolveSelection } from "../hooks/useSearchState";
import { plural } from "../lib/format";

export const EXAMPLES = [
  "a man is cooking pasta",
  "a cartoon character is dancing",
  "people playing basketball",
  "a woman applies makeup",
  "a car drives on a highway",
];

export function HomePage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const models = useModels();
  const collections = useCollections();
  const inputRef = useRef<HTMLInputElement>(null);
  useSlashFocus(inputRef);
  const { model, collection } = resolveSelection(params, models.data, collections.data);

  const go = (q: string) => navigate(`/search?${new URLSearchParams({ q, model, collection })}`);
  const pick = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    next.set(key, value);
    setParams(next, { replace: true });
  };
  const libraries = (collections.data ?? []).filter((c) => c.id !== "uploads");

  return (
    <>
      <section className="wrap-marketing pb-24 pt-20 text-center sm:pb-32 sm:pt-28">
        <p className="text-[19px] font-semibold text-accent-text">Spartan VLM</p>
        <h1 className="h-hero mt-3">Find the moment.</h1>
        <p className="lead mx-auto mt-5 max-w-[640px]">Describe what happens. Spartan finds the video, and the seconds that match.</p>
        <div className="mx-auto mt-10 max-w-[720px]">
          <SearchBar ref={inputRef} size="lg" onSubmit={go} />
          <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
            {models.data && models.data.length > 0 && (
              <ModelSwitch models={models.data} value={model} onChange={(v) => pick("model", v)} />
            )}
            {collections.data && collections.data.length > 0 && (
              <LibrarySelect collections={collections.data} value={collection} onChange={(v) => pick("collection", v)} />
            )}
          </div>
          <div className="mt-8">
            <p className="meta">Try an example</p>
            <ul className="mt-3 flex flex-wrap justify-center gap-2">
              {EXAMPLES.map((q) => (
                <li key={q}>
                  <button
                    type="button"
                    onClick={() => go(q)}
                    className="rounded-pill bg-surface px-4 py-2 text-[15px] shadow-sm ring-1 ring-hairline-soft transition-shadow duration-200 ease-out hover:shadow-md"
                  >
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="wrap-marketing pb-28 sm:pb-32" aria-labelledby="how">
        <h2 id="how" className="h-section max-w-[720px]">
          Search that understands what happens.
        </h2>
        <div className="mt-12 grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
          <FeatureCard n="01" title="Describe it" body="Type a plain sentence. No tags, no keywords, no filters.">
            <div className="flex items-center gap-2 rounded-pill bg-surface px-4 py-3 text-[15px] text-text-secondary shadow-sm">
              <span className="h-2 w-2 rounded-full bg-accent" aria-hidden />a dog catches a frisbee
            </div>
          </FeatureCard>
          <FeatureCard n="02" title="Ranked in one pass" body="Every clip is scored against your sentence, then the closest are checked again.">
            <div className="flex flex-col gap-2">
              {[91, 74, 58].map((v, i) => (
                <div key={v} className="flex items-center gap-3 text-[13px]">
                  <span className="w-4 text-text-secondary">{i + 1}</span>
                  <div className="h-2 flex-1 rounded-pill bg-fill">
                    <div className="h-2 rounded-pill bg-accent" style={{ width: `${v}%`, opacity: 1 - i * 0.25 }} />
                  </div>
                  <span className="w-10 text-right font-semibold text-accent-text">{v}%</span>
                </div>
              ))}
            </div>
          </FeatureCard>
          <FeatureCard n="03" title="Jump to the moment" body="The player opens at the seconds that match and loops them.">
            <div>
              <div className="relative h-2 rounded-pill bg-fill">
                <div className="absolute inset-y-0 left-[38%] w-[22%] rounded-pill bg-accent" />
              </div>
              <div className="mt-2 flex justify-between text-[12px] text-text-secondary">
                <span>0:00</span>
                <span className="font-semibold text-accent-text">0:05 – 0:08</span>
                <span>0:14</span>
              </div>
            </div>
          </FeatureCard>
        </div>
      </section>

      <section className="wrap-marketing pb-28 sm:pb-32" aria-labelledby="libraries">
        <h2 id="libraries" className="h-section">
          Pick a library.
        </h2>
        <div className="mt-12 grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
          {libraries.map((c) => (
            <Link
              key={c.id}
              to={`/library?collection=${c.id}`}
              className="card-lg group flex min-h-[200px] flex-col p-7 transition-shadow duration-200 ease-out hover:shadow-lg"
            >
              <h3 className="h-card">{c.name}</h3>
              <p className="mt-2 text-text-secondary">{c.description}</p>
              <p className="mt-auto flex items-center gap-1 pt-6 text-[15px] font-medium text-accent-text">
                {plural(c.video_count, "video")}
                <ChevronRight className="transition-transform duration-200 ease-out group-hover:translate-x-0.5" />
              </p>
            </Link>
          ))}
          <Link
            to="/upload"
            className="group flex min-h-[200px] flex-col rounded-card-lg bg-surface-dark p-7 text-text-on-dark shadow-md transition-shadow duration-200 ease-out hover:shadow-lg"
          >
            <h3 className="h-card">Your uploads</h3>
            <p className="mt-2 text-text-on-dark-secondary">Add your own video. It becomes searchable in a minute or two.</p>
            <p className="mt-auto flex items-center gap-1 pt-6 text-[15px] font-medium text-link-on-dark">
              Upload a video
              <ChevronRight className="transition-transform duration-200 ease-out group-hover:translate-x-0.5" />
            </p>
          </Link>
        </div>
      </section>
    </>
  );
}

function FeatureCard({ n, title, body, children }: { n: string; title: string; body: string; children: React.ReactNode }) {
  return (
    <div className="card-lg flex flex-col gap-5 p-7">
      <p className="text-[15px] font-semibold text-text-secondary">{n}</p>
      <div>
        <h3 className="h-card">{title}</h3>
        <p className="mt-2 text-text-secondary">{body}</p>
      </div>
      <div className="mt-auto rounded-[18px] bg-bg p-4">{children}</div>
    </div>
  );
}
