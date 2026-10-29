const DATA = window.EDA_DATA;
const COLORS = { msrvtt: "#0055a2", msvd: "#e5a823", train: "#0055a2", val: "#e5a823", test: "#6b7280" };
const PLOT_CONFIG = { displaylogo: false, responsive: true };

const fmt = (n, d = 0) => Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (a, b, d = 1) => (b ? (100 * a) / b : 0).toFixed(d) + "%";
const sum = (arr) => arr.reduce((a, b) => a + b, 0);

function layout(title, extra = {}) {
  return {
    title: { text: title, font: { size: 15 }, x: 0.02 },
    margin: { l: 55, r: 20, t: 45, b: 50 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "-apple-system, Segoe UI, Roboto, sans-serif", size: 12 },
    legend: { orientation: "h", y: -0.2 },
    bargap: 0.05,
    ...extra,
  };
}

function plot(id, traces, lay) {
  Plotly.react(id, traces, lay, PLOT_CONFIG);
}

function histTrace(h, name, color, opts = {}) {
  const x = h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2);
  const width = h.edges[1] - h.edges[0];
  return { type: "bar", x, y: h.counts, width, name, marker: { color }, ...opts };
}

function vline(x, label) {
  return {
    shapes: [{ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: "#dc2626", dash: "dash", width: 2 } }],
    annotations: [{ x, yref: "paper", y: 1, text: label, showarrow: false, xanchor: "right", font: { color: "#dc2626" } }],
  };
}

function table(el, headers, rows, numericFrom = 1) {
  const th = headers.map((h, i) => `<th class="${i >= numericFrom ? "num" : ""}">${h}</th>`).join("");
  const tr = rows
    .map((r) => "<tr>" + r.map((c, i) => `<td class="${i >= numericFrom ? "num" : ""}">${c}</td>`).join("") + "</tr>")
    .join("");
  document.getElementById(el).innerHTML = `<table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

function topResolution(v) {
  const [res, n] = Object.entries(v.resolutions)[0];
  return { res, share: pct(n, v.probed, 0) };
}

// ─── KPIs ──────────────────────────────────────────────────────────────────

function renderKpis(d) {
  const c = d.captions;
  const v = d.video;
  const top = topResolution(v);
  const cards = [
    ["Videos", fmt(d.videos), `${fmt(d.source_videos)} source YouTube videos`],
    ["Captions", fmt(c.total), `${fmt(c.per_video.mean, 1)} per video`],
    ["Avg duration", fmt(d.annotated_duration.mean, 1) + " s", `median ${fmt(d.annotated_duration.median, 1)} s`],
    ["Avg FPS", fmt(v.fps.mean, 1), `measured on ${fmt(v.probed)} videos`],
    ["Top resolution", top.res, `${top.share} of sample`],
    ["Words / caption", fmt(c.words.mean, 1), `max ${fmt(c.words.max)}`],
    ["> 77 CLIP tokens", c.pct_over_77_tokens.toFixed(2) + "%", `max ${fmt(c.clip_tokens.max)} tokens`],
    ["Vocabulary", fmt(c.vocab_size), "content words"],
  ];
  if (d.youtube) {
    const y = d.youtube;
    cards.push(["Still on YouTube", pct(y.clips_available, y.clips_total, 0), `checked ${y.checked_at}`]);
  }
  document.getElementById("kpis").innerHTML = cards
    .map(([l, val, note]) => `<div class="kpi"><div class="label">${l}</div><div class="value">${val}</div><div class="note">${note}</div></div>`)
    .join("");
}

// ─── Single-dataset view ───────────────────────────────────────────────────

function renderDataset(key) {
  const d = DATA.datasets[key];
  const color = COLORS[key];
  const c = d.captions;
  const v = d.video;
  renderKpis(d);

  // Splits
  const schemes = [...new Set(d.splits.map((s) => s.scheme))];
  plot(
    "chart-splits",
    schemes.map((sc, i) => {
      const rows = d.splits.filter((s) => s.scheme === sc);
      return { type: "bar", name: sc, x: rows.map((r) => r.split), y: rows.map((r) => r.videos), marker: { color: i ? "#9ca3af" : color } };
    }),
    layout("Videos per split", { barmode: "group", yaxis: { title: "videos" } })
  );
  table(
    "table-splits",
    ["Scheme", "Split", "Videos", "Captions", "Share of videos"],
    d.splits.map((s) => [s.scheme, s.split, fmt(s.videos), fmt(s.captions), pct(s.videos, d.videos)]),
    2
  );

  // Categories
  const catBlock = document.getElementById("categories-block");
  catBlock.hidden = !d.categories;
  if (d.categories) {
    const cat = d.categories;
    const names = [...cat.names].reverse();
    plot(
      "chart-categories",
      ["train", "val", "test"].map((s) => ({
        type: "bar", orientation: "h", name: s, y: names, x: [...cat[s]].reverse(), marker: { color: COLORS[s] },
      })),
      layout("Videos per category (official split)", { barmode: "stack", margin: { l: 150, r: 20, t: 45, b: 40 } })
    );
  }

  // Video properties
  plot(
    "chart-duration",
    [
      histTrace(d.annotated_duration.hist, "annotated (all videos)", color),
      { ...histTrace(v.duration.hist, "measured (sample)", "#9ca3af"), yaxis: "y2", opacity: 0.6 },
    ],
    layout("Clip duration (s)", {
      xaxis: { title: "seconds" }, yaxis: { title: "videos" },
      yaxis2: { overlaying: "y", side: "right", showgrid: false, title: "sample" }, barmode: "overlay",
    })
  );
  const fpsEntries = Object.entries(v.fps.counts).sort((a, b) => +a[0] - +b[0]);
  plot(
    "chart-fps",
    [{ type: "bar", x: fpsEntries.map((e) => e[0] + " fps"), y: fpsEntries.map((e) => e[1]), marker: { color } }],
    layout(`Frame rate (sample of ${fmt(v.probed)})`, { xaxis: { type: "category" }, yaxis: { title: "videos" } })
  );
  const res = Object.entries(v.resolutions);
  plot(
    "chart-resolution",
    [{ type: "bar", x: res.map((r) => r[0]), y: res.map((r) => r[1]), marker: { color } }],
    layout(`Resolution (${v.n_resolutions} distinct in sample)`, { xaxis: { type: "category", tickangle: -30 }, yaxis: { title: "videos" } })
  );
  const ar = Object.entries(v.aspect_ratios);
  plot(
    "chart-aspect",
    [{ type: "pie", hole: 0.5, labels: ar.map((a) => a[0]), values: ar.map((a) => a[1]), sort: false,
       marker: { colors: ["#0055a2", "#e5a823", "#10b981", "#8b5cf6", "#ef4444", "#9ca3af"] } }],
    layout("Aspect ratio (sample)", { legend: { orientation: "v" } })
  );

  // Captions
  plot("chart-words", [histTrace(c.words.hist, "captions", color)],
    layout("Words per caption (capped at 40)", { xaxis: { title: "words" }, yaxis: { title: "captions" }, ...vline(DATA.settings.min_words - 0.5, "pipeline min ") }));
  plot("chart-tokens", [histTrace(c.clip_tokens.hist, "captions", color)],
    layout("CLIP tokens per caption (incl. start/end)", { xaxis: { title: "tokens", range: [0, 80] }, yaxis: { title: "captions" }, ...vline(77, "CLIP limit 77 ") }));
  const tw = [...c.top_words].reverse();
  plot("chart-top-words", [{ type: "bar", orientation: "h", y: tw.map((w) => w[0]), x: tw.map((w) => w[1]), marker: { color } }],
    layout("Top 25 content words", { margin: { l: 90, r: 20, t: 45, b: 40 } }));
  const bg = [...c.top_bigrams].reverse();
  plot("chart-bigrams", [{ type: "bar", orientation: "h", y: bg.map((w) => w[0]), x: bg.map((w) => w[1]), marker: { color: "#10b981" } }],
    layout("Top 20 content bigrams", { margin: { l: 130, r: 20, t: 45, b: 40 } }));

  // Pixel stats
  const ch = ["R", "G", "B"];
  plot(
    "chart-rgb",
    [
      { type: "bar", name: "dataset mean", x: ch, y: v.rgb_mean, marker: { color },
        error_y: { type: "data", array: v.rgb_std, visible: true, color: "#374151" } },
      { type: "bar", name: "CLIP mean", x: ch, y: DATA.clip_norm.mean, marker: { color: "#9ca3af" },
        error_y: { type: "data", array: DATA.clip_norm.std, visible: true, color: "#374151" } },
    ],
    layout("Pixel mean ± std (0–1) vs CLIP normalization", { barmode: "group", yaxis: { range: [0, 0.9] } })
  );
  plot("chart-brightness", [histTrace(v.brightness_hist, "videos", color)],
    layout("Average frame brightness per video (sample)", { xaxis: { title: "mean intensity (0–1)" }, yaxis: { title: "videos" } }));

  // Quality
  const f = c.pipeline_filters;
  const afterLen = c.total - f.too_short - f.too_long - f.all_stopwords;
  const afterIntra = afterLen - f.intra_video_duplicates;
  plot(
    "chart-funnel",
    [{
      type: "funnel",
      y: ["Raw captions", `After length/stopword filter`, "After within-video dedup", "After cross-video dedup"],
      x: [c.total, afterLen, afterIntra, f.retained],
      texttemplate: "%{value:,} (%{percentInitial:.1%})",
      marker: { color: [color, color, color, "#dc2626"] },
    }],
    layout("Captions kept by the pipeline's text rules", { margin: { l: 190, r: 20, t: 45, b: 30 } })
  );
  if (d.youtube) {
    const splits = Object.entries(d.youtube.by_split);
    plot(
      "chart-youtube",
      [
        { type: "bar", name: "available", x: splits.map((s) => s[0]), y: splits.map((s) => s[1].available), marker: { color: "#10b981" } },
        { type: "bar", name: "removed / private", x: splits.map((s) => s[0]), y: splits.map((s) => s[1].total - s[1].available), marker: { color: "#dc2626" } },
      ],
      layout(`Source videos still on YouTube (checked ${d.youtube.checked_at})`, { barmode: "stack", yaxis: { title: "clips" } })
    );
  } else {
    document.getElementById("chart-youtube").innerHTML = "<p class='meta'>Run build_data.py with --check-youtube to populate.</p>";
  }
  table("table-repeated", ["Caption", "Occurrences"], Object.entries(c.examples.most_repeated).map(([k, n]) => [k, fmt(n)]));
}

// ─── Compare view ──────────────────────────────────────────────────────────

function renderCompare() {
  const ds = ["msrvtt", "msvd"].map((k) => [k, DATA.datasets[k]]);
  const rows = [
    ["Videos", (d) => fmt(d.videos)],
    ["Captions", (d) => fmt(d.captions.total)],
    ["Captions per video (mean / min / max)", (d) => `${fmt(d.captions.per_video.mean, 1)} / ${fmt(d.captions.per_video.min)} / ${fmt(d.captions.per_video.max)}`],
    ["Duration, median (s)", (d) => fmt(d.annotated_duration.median, 1)],
    ["Duration range (s)", (d) => `${fmt(d.annotated_duration.min, 1)} – ${fmt(d.annotated_duration.max, 1)}`],
    ["FPS, mean", (d) => fmt(d.video.fps.mean, 1)],
    ["Top resolution", (d) => `${topResolution(d.video).res} (${topResolution(d.video).share})`],
    ["Words per caption, mean", (d) => fmt(d.captions.words.mean, 1)],
    ["CLIP tokens, p95 / max", (d) => `${fmt(d.captions.clip_tokens.p95)} / ${fmt(d.captions.clip_tokens.max)}`],
    ["Captions > 77 tokens", (d) => d.captions.pct_over_77_tokens.toFixed(2) + "%"],
    ["Vocabulary (content words)", (d) => fmt(d.captions.vocab_size)],
    ["Captions kept by pipeline", (d) => d.captions.pipeline_filters.retained_pct + "%"],
    ["Clips still on YouTube", (d) => (d.youtube ? pct(d.youtube.clips_available, d.youtube.clips_total) : "–")],
  ];
  table("table-compare", ["Metric", "MSR-VTT", "MSVD"], rows.map(([l, f]) => [l, ...ds.map(([, d]) => f(d))]));

  const norm = (h) => ({ ...h, counts: h.counts.map((x) => x / sum(h.counts)) });
  const overlay = (id, title, pick, xtitle, extra = {}) =>
    plot(id, ds.map(([k, d]) => histTrace(norm(pick(d)), d.name, COLORS[k], { opacity: 0.6 })),
      layout(title, { barmode: "overlay", xaxis: { title: xtitle }, yaxis: { title: "share", tickformat: ".0%" }, ...extra }));

  overlay("cmp-duration", "Clip duration (s, 1 s bins)", (d) => d.annotated_duration.hist_1s, "seconds");
  overlay("cmp-tokens", "CLIP tokens per caption", (d) => d.captions.clip_tokens.hist, "tokens", { xaxis: { title: "tokens", range: [0, 60] } });
  overlay("cmp-words", "Words per caption", (d) => d.captions.words.hist, "words");
  plot("cmp-rgb",
    [...ds.map(([k, d]) => ({ type: "bar", name: d.name, x: ["R", "G", "B"], y: d.video.rgb_mean, marker: { color: COLORS[k] } })),
     { type: "bar", name: "CLIP", x: ["R", "G", "B"], y: DATA.clip_norm.mean, marker: { color: "#9ca3af" } }],
    layout("Pixel mean vs CLIP normalization", { barmode: "group" }));

  const kpis = ds.map(([, d]) => [
    [`${d.name} videos`, fmt(d.videos), `${fmt(d.captions.total)} captions`],
  ]).flat();
  document.getElementById("kpis").innerHTML = kpis
    .map(([l, val, note]) => `<div class="kpi"><div class="label">${l}</div><div class="value">${val}</div><div class="note">${note}</div></div>`)
    .join("");
}

// ─── Implications (derived from the data, not hardcoded) ───────────────────

function implications(keys) {
  const items = [];
  for (const k of keys) {
    const d = DATA.datasets[k];
    const c = d.captions;
    const v = d.video;
    const f = c.pipeline_filters;
    const n = d.name;
    items.push(
      `<b>${n} text length:</b> ${c.pct_over_77_tokens.toFixed(2)}% of captions exceed CLIP's 77-token limit ` +
      `(p95 = ${fmt(c.clip_tokens.p95)}, max = ${fmt(c.clip_tokens.max)}). Truncation almost never triggers, so it is not a concern.`
    );
    items.push(
      `<b>${n} frames:</b> median clip is ${fmt(d.annotated_duration.median, 1)} s at ~${fmt(v.fps.median)} fps ` +
      `(~${fmt(v.frames_per_video_native.median)} native frames). Sampling at 1 fps gives ~${fmt(v.frames_per_video_at_1fps.median)} frames per clip.`
    );
    if (v.fps.std < 0.05 && v.n_resolutions === 1) {
      items.push(
        `<b>${n} is re-encoded:</b> every sampled video in this copy is ${topResolution(v).res} at ${fmt(v.fps.mean)} fps, ` +
        `not the original YouTube resolution and frame rate. Frame sampling above ${fmt(v.fps.mean)} fps is impossible, ` +
        `which matters for temporal models (e.g. Spartan space-time attention) and event localization precision.`
      );
    }
    const top = topResolution(v);
    const ar = v.aspect_ratios;
    items.push(
      `<b>${n} resolution:</b> ${top.res} covers ${top.share} of the sample (${v.n_resolutions} distinct sizes; ` +
      Object.entries(ar).map(([a, x]) => `${a} ${pct(x, v.probed, 0)}`).join(", ") +
      `). Resize the short side to 224 and center-crop rather than squashing to 224×224.`
    );
    const gap = v.rgb_mean.map((m, i) => Math.abs(m - DATA.clip_norm.mean[i]));
    items.push(
      `<b>${n} pixels:</b> dataset RGB mean [${v.rgb_mean.map((x) => x.toFixed(3)).join(", ")}] vs CLIP ` +
      `[${DATA.clip_norm.mean.map((x) => x.toFixed(3)).join(", ")}] (max gap ${Math.max(...gap).toFixed(3)}). ` +
      `Use CLIP's normalization constants for the pretrained encoders.`
    );
    items.push(
      `<b>${n} pipeline filters:</b> ${fmt(f.too_short)} captions under ${DATA.settings.min_words} words, ` +
      `${fmt(f.intra_video_duplicates)} within-video duplicates, and <b>${fmt(f.cross_video_duplicates)} cross-video duplicates</b> ` +
      `(${pct(f.cross_video_duplicates, c.total)}). The cross-video dedup in <code>text_normalize.run_dedup</code> removes generic ` +
      `but valid captions (e.g. "${Object.keys(c.examples.most_repeated)[0]}") from all but one video.`
    );
    if (d.categories) {
      const totals = d.categories.names.map((_, i) => d.categories.train[i] + d.categories.val[i] + d.categories.test[i]);
      items.push(
        `<b>${n} categories:</b> ${d.categories.names[0]} (${fmt(totals[0])}) vs ${d.categories.names.at(-1)} ` +
        `(${fmt(totals.at(-1))}) is a ${(totals[0] / totals.at(-1)).toFixed(1)}× imbalance. Report per-category recall on the test set.`
      );
    }
    if (c.per_video.max - c.per_video.min > 5) {
      items.push(
        `<b>${n} caption counts vary</b> from ${fmt(c.per_video.min)} to ${fmt(c.per_video.max)} per video. ` +
        `Sample a fixed number of captions per video per epoch so heavily annotated clips don't dominate training.`
      );
    }
    if (d.youtube) {
      const y = d.youtube;
      items.push(
        `<b>${n} on YouTube:</b> only ${pct(y.clips_available, y.clips_total)} of clips are still available. ` +
        `Use the Hugging Face copy (${d.source.replace("https://huggingface.co/datasets/", "")}), not YouTube URLs.`
      );
    }
  }
  document.getElementById("implications").innerHTML = items.map((i) => `<li>${i}</li>`).join("");
}

// ─── Wiring ────────────────────────────────────────────────────────────────

function show(view) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  const compare = view === "compare";
  document.getElementById("dataset-view").hidden = compare;
  document.getElementById("compare-view").hidden = !compare;
  if (compare) {
    renderCompare();
    implications(["msrvtt", "msvd"]);
  } else {
    renderDataset(view);
    implications([view]);
  }
  window.dispatchEvent(new Event("resize"));
}

document.addEventListener("DOMContentLoaded", () => {
  if (!DATA) {
    document.querySelector("main").innerHTML = "<p>data.js not found. Run <code>python EDA/dashboard/build_data.py</code>.</p>";
    return;
  }
  const s = DATA.settings;
  const src = Object.values(DATA.datasets).map((d) => `<a href="${d.source}" target="_blank">${d.name}</a>`).join(", ");
  document.getElementById("meta").innerHTML =
    `Generated ${DATA.generated_at} from ${src}. Caption stats use every caption; video properties are measured on a ` +
    `random sample (seed ${s.seed}) of ${fmt(s.msrvtt_sample)} MSR-VTT and ${fmt(s.msvd_sample)} MSVD videos.`;
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      history.replaceState(null, "", "#" + b.dataset.view);
      show(b.dataset.view);
    })
  );
  const initial = location.hash.slice(1);
  show(["msrvtt", "msvd", "compare"].includes(initial) ? initial : "msrvtt");
});
