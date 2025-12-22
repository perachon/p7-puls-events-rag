import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def fetch_events(page_size: int = 50, after: str | None = None) -> dict:
    load_dotenv()

    base_url = os.getenv("OPENAGENDA_BASE_URL", "https://api.openagenda.com/v2")
    api_key = os.getenv("OPENAGENDA_API_KEY")
    agenda_uid = os.getenv("OPENAGENDA_AGENDA_UID")

    if not api_key:
        raise RuntimeError("OPENAGENDA_API_KEY manquante (mets-la dans .env).")
    if not agenda_uid:
        raise RuntimeError("OPENAGENDA_AGENDA_UID manquant (mets-le dans .env).")

    url = f"{base_url}/agendas/{agenda_uid}/events"

    headers = {
        "key": api_key,      # auth lecture via header "key"
        "lang": "fr",        # optionnel: simplifie les champs multilingues
    }

    params = {
        "size": page_size,   # taille de page (si supportée sur cette route)
    }
    if after:
        params["after"] = after  # curseur de pagination (si fourni par l’API)

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"OpenAgenda API error {r.status_code}: {r.text[:500]}"
        )

    return r.json()


def main():
    data = fetch_events(page_size=50)

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "openagenda_events_page1.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
