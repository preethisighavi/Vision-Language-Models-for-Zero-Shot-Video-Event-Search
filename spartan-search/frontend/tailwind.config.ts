import type { Config } from "tailwindcss";

const v = (name: string) => `var(--${name})`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: v("bg"),
        surface: v("surface"),
        "surface-dark": v("surface-dark"),
        video: v("video"),
        fill: v("fill"),
        track: v("track"),
        hairline: v("hairline"),
        "hairline-soft": v("hairline-soft"),
        text: v("text"),
        "text-secondary": v("text-secondary"),
        "text-nav": v("text-nav"),
        "text-on-dark": v("text-on-dark"),
        "text-on-dark-secondary": v("text-on-dark-secondary"),
        accent: v("accent"),
        "accent-text": v("accent-text"),
        "accent-hover": v("accent-hover"),
        "accent-tint": v("accent-tint"),
        "accent-ring": v("accent-ring"),
        "accent-secondary-span": v("accent-secondary-span"),
        "link-on-dark": v("link-on-dark"),
        "switch-on": v("switch-on"),
      },
      borderRadius: {
        pill: v("r-pill"),
        "card-lg": v("r-card-lg"),
        card: v("r-card"),
        video: v("r-video"),
        thumb: v("r-thumb"),
        "thumb-sm": v("r-thumb-sm"),
      },
      boxShadow: {
        sm: v("shadow-sm"),
        md: v("shadow-md"),
        lg: v("shadow-lg"),
        segment: v("shadow-segment"),
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', '"Inter"', '"Helvetica Neue"', "Arial", "sans-serif"],
      },
      maxWidth: { marketing: "1180px", app: "1240px" },
      transitionTimingFunction: { out: "cubic-bezier(0.2, 0.8, 0.2, 1)" },
    },
  },
  plugins: [],
} satisfies Config;
