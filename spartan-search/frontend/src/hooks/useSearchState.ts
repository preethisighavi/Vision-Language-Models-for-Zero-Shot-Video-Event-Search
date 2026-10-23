import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";

import type { CollectionInfo, ModelInfo } from "../api/client";

/** Picks the model and library from the URL, falling back to sensible defaults. */
export function resolveSelection(
  params: URLSearchParams,
  models: ModelInfo[] | undefined,
  collections: CollectionInfo[] | undefined,
) {
  const modelParam = params.get("model");
  const model =
    (models ?? []).find((m) => m.id === modelParam)?.id ??
    (models ?? []).find((m) => m.default)?.id ??
    models?.[0]?.id ??
    "";
  const collectionParam = params.get("collection");
  const collection =
    (collections ?? []).find((c) => c.id === collectionParam)?.id ??
    (collections ?? []).find((c) => c.video_count > 0)?.id ??
    collections?.[0]?.id ??
    "";
  return { model, collection };
}

export function useUrlState() {
  const [params, setParams] = useSearchParams();
  return useMemo(() => {
    const update = (patch: Record<string, string | null | undefined>, replace = false) => {
      const next = new URLSearchParams(params);
      for (const [k, v] of Object.entries(patch)) {
        if (v === null || v === undefined || v === "") next.delete(k);
        else next.set(k, v);
      }
      setParams(next, { replace });
    };
    return { params, update };
  }, [params, setParams]);
}
