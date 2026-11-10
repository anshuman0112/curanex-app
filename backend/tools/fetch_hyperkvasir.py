"""Fetches HyperKvasir (Borgli et al. 2020, Baerum Hospital, Norway; OSF mh9sj) folders used as an outside test for the
endoscopy and oesophagitis models. Images are fetched one by one from OSF. Run from backend/."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "hyperkvasir" / "images"
BASE = "https://api.osf.io/v2/nodes/mh9sj/files/osfstorage/"
FOLDERS = {
    "z-line": "5dfb5b9bfad251000a69ef75",
    "esophagitis-a": "5dfb5b05fad251000a69ee79",
    "esophagitis-b-d": "5dfb5b140236b8000a73911a",
    "cecum": "5dfb5b530236b8000c746a8e",
    "polyps": "5dfb5ba6fad251000c6a3034",
    "ulcerative-colitis-grade-0-1": "5dfb5c34a95d73000b3a4d52",
    "ulcerative-colitis-grade-1-2": "5dfb5c4afad251000a69f06a",
    "ulcerative-colitis-grade-2-3": "5dfb5c560236b8000c746cba",
    "ulcerative-colitis-grade-1": "5dfb5c6bfad251000b6a9b80",
    "ulcerative-colitis-grade-2": "5dfb5c740236b8000c746d21",
    "ulcerative-colitis-grade-3": "5dfb5c7b0236b8000b73e5fd",
}
LIMIT = 600  # per folder


def listing(client: httpx.Client, folder_id: str) -> list[tuple[str, str]]:
    out, url = [], f"{BASE}{folder_id}/"
    while url and len(out) < LIMIT:
        d = client.get(url, timeout=60).json()
        out += [(f["attributes"]["name"], f["links"]["download"]) for f in d["data"] if f["attributes"]["kind"] == "file"]
        url = d["links"].get("next")
    return out[:LIMIT]


def main() -> None:
    with httpx.Client(follow_redirects=True) as client:
        jobs = []
        for name, fid in FOLDERS.items():
            (OUT / name).mkdir(parents=True, exist_ok=True)
            files = listing(client, fid)
            jobs += [(OUT / name / n, u) for n, u in files if not (OUT / name / n).exists()]
            print(name, len(files), flush=True)

        def get(job):
            path, url = job
            for _ in range(3):
                try:
                    r = client.get(url, timeout=120)
                    if r.status_code == 200:
                        path.write_bytes(r.content)
                        return 1
                except httpx.HTTPError:
                    pass
            return 0

        with ThreadPoolExecutor(8) as pool:
            done = sum(pool.map(get, jobs))
    print("downloaded", done, "of", len(jobs), file=sys.stderr)


if __name__ == "__main__":
    main()
