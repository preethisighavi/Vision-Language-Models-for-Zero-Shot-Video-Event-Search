import type { components } from "./schema";

type S = components["schemas"];
export type ModelInfo = S["ModelInfo"];
export type CollectionInfo = S["CollectionInfo"];
export type SearchRequest = S["SearchRequest"];
export type SearchResponse = S["SearchResponse"];
export type SearchResult = S["SearchResult"];
export type Span = S["Span"];
export type VideoDetail = S["VideoDetail"];
export type VideoSummary = S["VideoSummary"];
export type VideoPage = S["VideoPage"];
export type LocalizeResponse = S["LocalizeResponse"];
export type CompareRequest = S["CompareRequest"];
export type CompareResponse = S["CompareResponse"];
export type UploadAccepted = S["UploadAccepted"];
export type JobInfo = S["JobInfo"];
export type BenchmarkRow = S["BenchmarkRow"];
export type FeedbackRequest = S["FeedbackRequest"];
export type Health = S["Health"];

export interface AskResponse {
  video_id: string;
  model: string;
  query: string;
  answer: string;
  spans: Span[];
  saliency: number[];
  saliency_fps: number;
  latency_ms: number;
}

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api/v1";

/** Media URLs from the API are root-relative ("/api/v1/..."). Resolve them against the API origin. */
const API_ORIGIN = /^https?:\/\//.test(API_BASE) ? new URL(API_BASE).origin : "";

export function mediaUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_ORIGIN}${path}`;
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: init.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init.headers },
    });
  } catch {
    throw new ApiError(0, "network_error", "Can't reach the server. Check your connection and try again.");
  }
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = body?.error;
    throw new ApiError(res.status, err?.code ?? "http_error", err?.message ?? `Request failed (${res.status}).`);
  }
  return body as T;
}

const post = <T>(path: string, data: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(data) });

export const api = {
  health: () => request<Health>("/health"),
  models: () => request<ModelInfo[]>("/models"),
  collections: () => request<CollectionInfo[]>("/collections"),
  search: (req: Partial<SearchRequest> & Pick<SearchRequest, "query" | "model" | "collection">) =>
    post<SearchResponse>("/search", req),
  video: (id: string) => request<VideoDetail>(`/videos/${encodeURIComponent(id)}`),
  videos: (collection: string | undefined, page = 1, pageSize = 24) => {
    const q = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (collection) q.set("collection", collection);
    return request<VideoPage>(`/videos?${q}`);
  },
  localize: (id: string, query: string, model: string) =>
    post<LocalizeResponse>(`/videos/${encodeURIComponent(id)}/localize`, { query, model }),
  ask: (id: string, query: string, model: string) =>
    post<AskResponse>(`/videos/${encodeURIComponent(id)}/ask`, { query, model }),
  compare: (req: CompareRequest) => post<CompareResponse>("/search/compare", req),
  upload: (file: File, title?: string, uploadKey?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (title) form.append("title", title);
    return request<UploadAccepted>("/videos", {
      method: "POST",
      body: form,
      headers: uploadKey ? { "X-Upload-Key": uploadKey } : undefined,
    });
  },
  job: (id: string) => request<JobInfo>(`/jobs/${encodeURIComponent(id)}`),
  benchmarks: () => request<BenchmarkRow[]>("/benchmarks"),
  feedback: (req: FeedbackRequest) => post<{ status: string }>("/feedback", req),
};

export const streamUrl = (videoId: string) => mediaUrl(`${API_BASE.replace(/^https?:\/\/[^/]+/, "")}/videos/${videoId}/stream`);
export const thumbUrl = (videoId: string, t = 0) =>
  mediaUrl(`${API_BASE.replace(/^https?:\/\/[^/]+/, "")}/videos/${videoId}/thumbnail?t=${t.toFixed(1)}`);
