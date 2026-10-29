import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { SearchResult } from "../api/client";
import { RankChange } from "../components/Compare";
import { ModelSwitch, RerankSwitch, SearchBar } from "../components/Controls";
import { Player } from "../components/Player";
import { ResultCard } from "../components/ResultCard";

const result: SearchResult = {
  video_id: "video7020",
  title: "a man is cooking pasta",
  collection: "msrvtt-1ka",
  rank: 1,
  score: 0.91,
  stage1_score: 0.34,
  rerank_score: 2.7,
  duration: 14.2,
  thumbnail_url: "/api/v1/videos/video7020/thumbnail?t=5.3",
  stream_url: "/api/v1/videos/video7020/stream",
  spans: [
    { start: 5.3, end: 11, score: 0.82 },
    { start: 12, end: 13.5, score: 0.4 },
  ],
  saliency: [0.1, 0.4, 0.9],
  saliency_fps: 1,
};

function withQuery(ui: React.ReactElement) {
  return <QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>;
}

describe("ResultCard", () => {
  it("shows rank, match, span and selection state", async () => {
    const onSelect = vi.fn();
    render(
      <ul>
        <ResultCard result={result} selected={false} collectionName="MSR-VTT 1k-A" onSelect={onSelect} />
      </ul>,
    );
    const button = screen.getByRole("button", { name: /a man is cooking pasta/ });
    expect(button).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("1. 91% match")).toBeInTheDocument();
    expect(screen.getByText("0:05 – 0:11")).toBeInTheDocument();
    expect(screen.getByText("MSR-VTT 1k-A · 0:14")).toBeInTheDocument();
    await userEvent.click(button);
    expect(onSelect).toHaveBeenCalled();
  });
});

describe("Controls", () => {
  it("submits the trimmed query on Enter", async () => {
    const onSubmit = vi.fn();
    render(<SearchBar onSubmit={onSubmit} />);
    await userEvent.type(screen.getByRole("searchbox"), "  dog park {enter}");
    expect(onSubmit).toHaveBeenCalledWith("dog park");
  });

  it("marks the selected model with aria-pressed", async () => {
    const onChange = vi.fn();
    const models = [
      { id: "a", name: "Model A", description: "", embedding_dim: 512, supports_rerank: true, supports_localize: true, default: true, status: "ready" as const },
      { id: "b", name: "Model B", description: "", embedding_dim: 512, supports_rerank: false, supports_localize: true, default: false, status: "ready" as const },
    ];
    render(<ModelSwitch models={models} value="a" onChange={onChange} />);
    expect(screen.getByRole("button", { name: "Model A" })).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(screen.getByRole("button", { name: "Model B" }));
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("re-rank is a real switch", async () => {
    const onChange = vi.fn();
    render(<RerankSwitch checked onChange={onChange} />);
    const sw = screen.getByRole("switch", { name: "Re-rank" });
    expect(sw).toHaveAttribute("aria-checked", "true");
    await userEvent.click(sw);
    expect(onChange).toHaveBeenCalledWith(false);
  });
});

describe("Player", () => {
  it("loops the selected span and moves between moments with arrow keys", () => {
    render(withQuery(<Player videoId="v" title="Clip" streamUrl="/s.mp4" duration={14.2} spans={result.spans} queryId="q_1" />));
    const video = screen.getByLabelText("Video: Clip") as HTMLVideoElement;
    expect(screen.getByText("Playing 0:05 – 0:11 · looping")).toBeInTheDocument();

    // Past the end of the span: the player jumps back to its start.
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 11.5 });
    fireEvent.timeUpdate(video);
    expect(video.currentTime).toBe(5.3);

    const first = screen.getByRole("button", { name: "0:05 – 0:11 · 82%" });
    first.focus();
    fireEvent.keyDown(first, { key: "ArrowRight" });
    expect(screen.getByRole("button", { name: "0:12 – 0:13 · 40%" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Playing 0:12 – 0:13 · looping")).toBeInTheDocument();
  });

  it("stops looping once the user scrubs", () => {
    render(withQuery(<Player videoId="v" title="Clip" streamUrl="/s.mp4" duration={14.2} spans={result.spans} />));
    const video = screen.getByLabelText("Video: Clip");
    fireEvent.seeking(video); // not triggered by the player
    expect(screen.getByText("Playing 0:05 – 0:11")).toBeInTheDocument();
  });

  it("asks for feedback and thanks the user", async () => {
    render(withQuery(<Player videoId="v" title="Clip" streamUrl="/s.mp4" duration={14.2} spans={result.spans} queryId="q_1" />));
    await userEvent.click(screen.getByRole("button", { name: "Yes" }));
    expect(screen.getByText("Thanks. Your answer helps us evaluate the model.")).toBeInTheDocument();
  });
});

describe("RankChange", () => {
  it("shows up, down, same and new", () => {
    render(
      <div>
        <RankChange change={2} />
        <RankChange change={-1} />
        <RankChange change={0} />
        <RankChange change="new" />
      </div>,
    );
    expect(screen.getByLabelText("Up 2")).toHaveTextContent("↑ 2");
    expect(screen.getByLabelText("Down 1")).toHaveTextContent("↓ 1");
    expect(screen.getByLabelText("Same rank")).toHaveTextContent("–");
    expect(screen.getByText("new")).toBeInTheDocument();
  });
});
