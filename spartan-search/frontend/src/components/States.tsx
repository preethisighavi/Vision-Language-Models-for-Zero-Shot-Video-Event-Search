import type { ReactNode } from "react";

import { ApiError } from "../api/client";
import { AlertIcon, ClockIcon } from "./Icons";

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="card flex flex-col items-center gap-3 px-6 py-14 text-center">
      <h2 className="text-[21px] font-semibold tracking-[-0.015em]">{title}</h2>
      {children && <div className="max-w-md text-text-secondary">{children}</div>}
      {action}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof ApiError || error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div role="alert" className="card flex flex-col items-center gap-3 px-6 py-12 text-center">
      <AlertIcon className="text-text-secondary" size={28} />
      <h2 className="text-[21px] font-semibold tracking-[-0.015em]">That didn't work.</h2>
      <p className="max-w-md text-text-secondary">{message}</p>
      {onRetry && (
        <button type="button" className="btn-secondary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function IndexBuildingBanner({ message }: { message: string }) {
  return (
    <div role="status" className="flex items-start gap-3 rounded-card bg-surface px-5 py-4 shadow-sm">
      <ClockIcon className="mt-0.5 shrink-0 text-accent" />
      <div>
        <p className="font-semibold">This library isn't ready for this model yet.</p>
        <p className="text-[15px] text-text-secondary">{message} Pick another model or library, or check back after the index is built.</p>
      </div>
    </div>
  );
}

export function isIndexNotReady(error: unknown): error is ApiError {
  return error instanceof ApiError && error.status === 409;
}

export function ResultSkeleton() {
  return (
    <li className="card flex gap-4 p-[14px]" aria-hidden>
      <div className="skeleton aspect-video w-[132px] shrink-0 rounded-thumb sm:w-[184px]" />
      <div className="flex flex-1 flex-col gap-2 py-1">
        <div className="skeleton h-3 w-24 rounded" />
        <div className="skeleton h-4 w-3/4 rounded" />
        <div className="skeleton h-3 w-1/2 rounded" />
        <div className="skeleton mt-auto h-1 w-full rounded" />
      </div>
    </li>
  );
}

export function PlayerSkeleton() {
  return (
    <div className="card-lg p-4" aria-hidden>
      <div className="skeleton aspect-video w-full rounded-video" />
      <div className="skeleton mt-4 h-5 w-2/3 rounded" />
      <div className="skeleton mt-3 h-2 w-full rounded" />
    </div>
  );
}
