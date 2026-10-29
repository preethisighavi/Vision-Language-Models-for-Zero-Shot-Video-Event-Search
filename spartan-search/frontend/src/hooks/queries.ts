import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { useEffect } from "react";

import { api, type CompareRequest, type FeedbackRequest } from "../api/client";

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: api.models, staleTime: 60_000 });
}

export function useCollections() {
  return useQuery({ queryKey: ["collections"], queryFn: api.collections, staleTime: 15_000 });
}

export interface SearchParams {
  query: string;
  model: string;
  collection: string;
  rerank: boolean;
  topK?: number;
  videoId?: string;
}

export function useSearch(p: SearchParams | null) {
  return useQuery({
    queryKey: ["search", p],
    enabled: !!p && !!p.query && !!p.model && !!p.collection,
    queryFn: () =>
      api.search({
        query: p!.query,
        model: p!.model,
        collection: p!.collection,
        rerank: p!.rerank,
        top_k: p!.topK ?? 10,
        with_spans: true,
        video_id: p!.videoId ?? null,
      }),
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: 30_000,
  });
}

export function useVideo(id: string | undefined) {
  return useQuery({ queryKey: ["video", id], enabled: !!id, queryFn: () => api.video(id!), retry: false });
}

export function useVideos(collection: string | undefined, page: number, pageSize = 24) {
  return useQuery({
    queryKey: ["videos", collection, page, pageSize],
    queryFn: () => api.videos(collection, page, pageSize),
    placeholderData: keepPreviousData,
  });
}

export function useLocalize(videoId: string | undefined, query: string, model: string) {
  return useQuery({
    queryKey: ["localize", videoId, query, model],
    enabled: !!videoId && !!query && !!model,
    queryFn: () => api.localize(videoId!, query, model),
    retry: false,
  });
}

export function useCompare(req: CompareRequest | null) {
  return useQuery({
    queryKey: ["compare", req],
    enabled: !!req && !!req.query && req.models.length >= 2,
    queryFn: () => api.compare(req!),
    retry: false,
    placeholderData: keepPreviousData,
  });
}

export function useBenchmarks() {
  return useQuery({ queryKey: ["benchmarks"], queryFn: api.benchmarks, staleTime: 60_000 });
}

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: ["job", jobId],
    enabled: !!jobId,
    queryFn: () => api.job(jobId!),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "indexed" || status === "failed" ? false : 2000;
    },
  });
}

export function useFeedback() {
  return useMutation({ mutationFn: (req: FeedbackRequest) => api.feedback(req) });
}

/** "/" focuses the main search input from anywhere (unless the user is typing). */
export function useSlashFocus(ref: React.RefObject<HTMLInputElement>) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      e.preventDefault();
      ref.current?.focus();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ref]);
}
