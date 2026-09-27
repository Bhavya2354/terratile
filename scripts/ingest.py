import argparse
from collections import Counter
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="POST every PNG in a folder to a running terratile service.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    outcomes: Counter[str] = Counter()
    with httpx.Client(base_url=args.url, timeout=30) as client:
        for path in sorted(args.folder.glob("*.png")):
            with path.open("rb") as f:
                r = client.post("/tiles", files={"file": (path.name, f, "image/png")},
                                data={"source": args.source} if args.source else None)
            if r.status_code >= 400:
                outcomes["error"] += 1
                print(f"{path.name}: {r.status_code} {r.text}")
                continue
            body = r.json()
            outcomes["duplicate" if r.status_code == 200 else body["status"]] += 1
    print(dict(outcomes))


if __name__ == "__main__":
    main()
