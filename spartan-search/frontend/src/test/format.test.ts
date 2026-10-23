import { formatMatch, formatMetric, formatSpan, formatTime, matchPercent } from "../lib/format";

describe("format", () => {
  it("formats times as m:ss", () => {
    expect(formatTime(0)).toBe("0:00");
    expect(formatTime(5.9)).toBe("0:05");
    expect(formatTime(65.4)).toBe("1:05");
    expect(formatTime(-3)).toBe("0:00");
  });

  it("formats spans with an en dash and spaces", () => {
    expect(formatSpan({ start: 5.3, end: 11 })).toBe("0:05 – 0:11");
  });

  it("shows scores as an integer match percentage", () => {
    expect(matchPercent(0.913)).toBe(91);
    expect(matchPercent(1.4)).toBe(100);
    expect(formatMatch(0.5)).toBe("50% match");
  });

  it("formats benchmark metrics", () => {
    expect(formatMetric("R@1", 0.3664)).toBe("36.6%");
    expect(formatMetric("MRR", 0.50668)).toBe("0.507");
    expect(formatMetric("latency_ms", 36.5)).toBe("37 ms");
    expect(formatMetric("MedR", 2)).toBe("2");
  });
});
