import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def fetch_page(agenda_uid: str, api_key: str, base_url: str, size: int = 100, after=None) -> dict:
    url = f"{base_url}/agendas/{agenda_uid}/events"
    headers = {"key": api_key, "lang": "fr"}
    params = {"size": size}
    if after is not None:
        params["after"] = after

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAgenda API error {r.status_code}: {r.text[:500]}")
    return r.json()


def main():
    load_dotenv()
    base_url = os.getenv("OPENAGENDA_BASE_URL", "https://api.openagenda.com/v2")
    api_key = os.getenv("OPENAGENDA_API_KEY")
    agenda_uid = os.getenv("OPENAGENDA_AGENDA_UID")

    if not api_key or not agenda_uid:
        raise RuntimeError("OPENAGENDA_API_KEY ou OPENAGENDA_AGENDA_UID manquant dans .env")

    size = int(os.getenv("OPENAGENDA_PAGE_SIZE", "100"))

    all_events = []
    after = None
    page = 0

    while True:
        page += 1
        data = fetch_page(agenda_uid, api_key, base_url, size=size, after=after)

        events = data.get("events") or data.get("data") or []
        if not events:
            break

        all_events.extend(events)
        after = data.get("after")  # pagination par curseur
        print(f"Page {page}: +{len(events)} (total={len(all_events)})")

        if after is None:
            break

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "openagenda_events_all.json"
    out_path.write_text(json.dumps({"events": all_events}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} ({len(all_events)} events)")


if __name__ == "__main__":
    main()
