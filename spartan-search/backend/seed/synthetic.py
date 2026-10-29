"""30 short labelled test clips, so the app works with zero downloads."""

import csv
from pathlib import Path

from app.services import media

MSRVTT = [
    "a man is cooking pasta in a kitchen",
    "a chef slices vegetables on a cutting board",
    "a cartoon character is dancing on stage",
    "people playing basketball in a gym",
    "a woman applies makeup in front of a mirror",
    "a car drives on a highway at sunset",
    "a dog catches a frisbee in the park",
    "a man plays the guitar on a sofa",
    "kids swimming in a pool",
    "a news anchor talks about the weather",
    "a soccer player scores a goal",
    "a woman is singing on stage",
    "a cat plays with a ball of yarn",
    "a man rides a horse through a field",
    "people dancing at a wedding party",
    "a girl is drawing a picture with crayons",
    "a tennis player serves the ball",
    "a man fixes a car engine in a garage",
]
MSVD = [
    "a woman is slicing an onion",
    "a monkey is eating a banana",
    "a man is riding a bicycle",
    "a dog is running on the beach",
    "a woman is pouring milk into a bowl",
    "a boy is playing the piano",
    "a man is lifting weights",
    "a baby is laughing",
    "a person is folding paper",
    "a squirrel is eating a peanut",
    "a man is cutting a tomato",
    "two men are boxing in a ring",
]
COLORS = ["0x0a66e0", "0xe0610a", "0x2e9e4f", "0x8e44ad", "0xc0392b", "0x16a085", "0xd4ac0d", "0x34495e"]
PATTERNS = ["testsrc2", "testsrc", "rgbtestsrc", "smptehdbars"]


def generate(seed_dir: Path, log=print) -> int:
    """Writes clips + metadata.csv into seed_dir/<collection>/. Returns the number of clips."""
    rows = []
    plan = [("msrvtt-1ka", f"video{7010 + i}", t) for i, t in enumerate(MSRVTT)]
    plan += [("msvd-test", f"msvd{100 + i}", t) for i, t in enumerate(MSVD)]
    for i, (collection, video_id, title) in enumerate(plan):
        dest = seed_dir / collection / f"{video_id}.mp4"
        duration = 8 + (i * 7) % 17  # 8-24 s
        if not dest.exists():
            log(f"  generating {collection}/{video_id}.mp4 ({duration}s) - {title}")
            media.make_test_video(dest, title, duration, COLORS[i % len(COLORS)], PATTERNS[i % len(PATTERNS)])
        rows.append({
            "video_id": video_id,
            "title": title,
            "collection": collection,
            "source": "Synthetic test clip (ffmpeg)",
            "license": "CC0",
            "captions": title,
        })
    for collection in {r["collection"] for r in rows}:
        path = seed_dir / collection / "metadata.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(r for r in rows if r["collection"] == collection)
    return len(rows)
