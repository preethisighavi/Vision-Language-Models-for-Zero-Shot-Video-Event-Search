import { useEffect, useRef, useState } from "react";

import { api, ApiError, type Span, streamUrl } from "../api/client";
import { ModelSwitch } from "./Controls";
import { Player } from "./Player";
import { ErrorState, IndexBuildingBanner } from "./States";
import { useModels, useVideo } from "../hooks/queries";
import { formatTime } from "../lib/format";

interface Turn {
  id: number;
  query: string;
  modelId: string;
  modelName: string;
  answer?: string;
  spans?: Span[];
  saliency?: number[] | null;
  error?: string;
  errorCode?: string;
  pending?: boolean;
}

/** Ask questions about one video. Each answer is the moment in the file that matches. */
export function VideoAsk({ videoId, initialModel }: { videoId: string; initialModel?: string }) {
  const video = useVideo(videoId);
  const models = useModels();
  const list = (models.data ?? []).filter((m) => m.supports_localize);
  const [model, setModel] = useState(initialModel ?? "");
  const chosen = list.find((m) => m.id === model) ?? list.find((m) => m.default) ?? list[0];
  const [turns, setTurns] = useState<Turn[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [asking, setAsking] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  const active = turns.find((t) => t.id === activeId) ?? [...turns].reverse().find((t) => t.spans);
  const v = video.data;

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  async function ask(query: string) {
    if (!chosen || asking) return;
    const id = nextId.current++;
    const turn: Turn = { id, query, modelId: chosen.id, modelName: chosen.name, pending: true };
    setTurns((t) => [...t, turn]);
    setActiveId(id);
    setAsking(true);
    setDraft("");
    try {
      const res = await api.ask(videoId, query, chosen.id);
      setTurns((t) =>
        t.map((item) =>
          item.id === id
            ? { ...item, pending: false, answer: res.answer, spans: res.spans, saliency: res.saliency }
            : item,
        ),
      );
    } catch (e) {
      const message = e instanceof ApiError ? e.message : "That question failed. Try again.";
      const errorCode = e instanceof ApiError ? e.code : "http_error";
      setTurns((t) => t.map((item) => (item.id === id ? { ...item, pending: false, error: message, errorCode } : item)));
    } finally {
      setAsking(false);
    }
  }

  if (video.error) return <ErrorState error={video.error} />;
  if (!v) return <div className="card skeleton h-[420px]" aria-hidden />;

  return (
    <div className="flex flex-wrap items-start gap-6">
      <section className="flex min-w-0 flex-col" style={{ flex: "999 1 420px" }} aria-label="Questions">
        <div className="flex flex-wrap items-center gap-3">
          {list.length > 0 && (
            <ModelSwitch models={list} value={chosen?.id ?? ""} onChange={setModel} label="Model" />
          )}
          {chosen && <p className="meta">Answers come from {chosen.name}.</p>}
        </div>

        <div ref={threadRef} className="mt-4 flex max-h-[520px] flex-col gap-3 overflow-y-auto pr-1" aria-live="polite">
          {turns.length === 0 && (
            <p className="card px-5 py-8 text-text-secondary">
              Ask a question about this video. The answer is the moment that matches, and the player opens there.
            </p>
          )}
          {turns.map((t) => (
            <article key={t.id} className="flex flex-col gap-2">
              <p className="ml-auto max-w-[85%] rounded-card bg-fill px-4 py-3">{t.query}</p>
              <div className="max-w-[85%] rounded-card bg-surface px-4 py-3 shadow-sm">
                <p className="text-[12px] font-semibold text-text-secondary">{t.modelName}</p>
                {t.pending ? (
                  <p className="mt-1 text-text-secondary">Looking through the video…</p>
                ) : t.error ? (
                  t.errorCode === "index_not_ready" ? (
                    <div className="mt-2">
                      <IndexBuildingBanner message={t.error} />
                    </div>
                  ) : (
                    <p className="mt-1">{t.error}</p>
                  )
                ) : (
                  <>
                    <p className="mt-1">{t.answer}</p>
                    {t.spans && t.spans.length > 0 && (
                      <button
                        type="button"
                        className="mt-2 text-[14px] font-medium text-accent-text hover:underline"
                        onClick={() => setActiveId(t.id)}
                      >
                        Play this moment
                      </button>
                    )}
                  </>
                )}
              </div>
            </article>
          ))}
        </div>

        <form
          className="mt-4 flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            const q = draft.trim();
            if (q) ask(q.slice(0, 200));
          }}
        >
          <label htmlFor={`ask-${videoId}`} className="sr-only">
            Ask a question about this video
          </label>
          <input
            id={`ask-${videoId}`}
            value={draft}
            maxLength={200}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask about this video"
            className="h-[52px] min-w-0 flex-1 rounded-pill bg-surface px-5 shadow-lg ring-1 ring-hairline-soft focus:outline-none focus:ring-2 focus:ring-accent"
          />
          <button type="submit" className="btn-primary h-[52px] px-6" disabled={asking || !draft.trim() || !chosen}>
            Ask
          </button>
        </form>
      </section>

      <div className="min-w-0 lg:sticky lg:top-[72px]" style={{ flex: "1 1 440px" }}>
        <Player
          videoId={v.id}
          title={v.title}
          streamUrl={streamUrl(v.id)}
          duration={v.duration}
          score={active?.spans?.[0]?.score}
          meta={`${formatTime(v.duration)}${chosen ? ` · ${chosen.name}` : ""}`}
          spans={active?.spans ?? []}
          saliency={active?.saliency}
          modelName={active?.modelName ?? chosen?.name}
          showFeedback={false}
        />
      </div>
    </div>
  );
}
