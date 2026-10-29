import { useState } from "react";

import { PlayGlyph } from "./Icons";

/** A lazy thumbnail on the dark video placeholder. Never blocks layout. */
export function Thumb({
  src,
  alt = "",
  className = "",
  rounded = "rounded-thumb",
  showPlay = true,
  children,
}: {
  src: string;
  alt?: string;
  className?: string;
  rounded?: string;
  showPlay?: boolean;
  children?: React.ReactNode;
}) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <div className={`relative aspect-video overflow-hidden bg-video ${rounded} ${className}`}>
      {!failed && (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          className={`h-full w-full object-cover transition-opacity duration-200 ease-out ${loaded ? "opacity-100" : "opacity-0"}`}
        />
      )}
      {showPlay && (
        <span className="absolute inset-0 grid place-items-center" aria-hidden>
          <span className="grid h-9 w-9 place-items-center rounded-full bg-black/45 text-white backdrop-blur-sm">
            <PlayGlyph size={16} />
          </span>
        </span>
      )}
      {children}
    </div>
  );
}
