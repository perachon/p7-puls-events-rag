import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


# RAW_PATH = Path("data/raw/openagenda_events_page1.json")
RAW_PATH = Path("data/raw/openagenda_events_all.json")
OUT_DIR = Path("data/processed")


def _get(d: dict, path: str, default=None):
    """Safe nested getter: path like 'firstTiming.begin'."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_dt(s: str | None):
    if not s:
        return pd.NaT
    # OpenAgenda dates are ISO-8601 with timezone offset, e.g. "2025-03-07T00:00:00+01:00"
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_raw_events(raw_path: Path) -> list[dict]:
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    # In your export, events are in "events" and "total" at root.
    events = data.get("events") or data.get("data") or []
    if not isinstance(events, list):
        raise ValueError("Format JSON inattendu: 'events'/'data' n'est pas une liste.")
    return events


def to_dataframe(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        row = {
            "uid": ev.get("uid"),
            "title_fr": _get(ev, "title.fr"),
            "description_fr": _get(ev, "description.fr"),
            "keywords_fr": _get(ev, "keywords.fr", []),
            "thematique": _get(ev, "thematique", []),
            "type_devenement": _get(ev, "type-devenement"),

            "first_begin": _get(ev, "firstTiming.begin"),
            "first_end": _get(ev, "firstTiming.end"),
            "last_begin": _get(ev, "lastTiming.begin"),
            "last_end": _get(ev, "lastTiming.end"),

            "location_name": _get(ev, "location.name"),
            "location_address": _get(ev, "location.address"),
            "location_city": _get(ev, "location.city"),
            "location_postal": _get(ev, "location.postalCode"),
            "location_lat": _get(ev, "location.latitude"),
            "location_lon": _get(ev, "location.longitude"),

            "origin_url": ev.get("originalUrl") or ev.get("url"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Parse datetimes
    df["first_begin_dt"] = df["first_begin"].apply(_parse_dt)
    df["first_end_dt"] = df["first_end"].apply(_parse_dt)
    df["last_begin_dt"] = df["last_begin"].apply(_parse_dt)
    df["last_end_dt"] = df["last_end"].apply(_parse_dt)

    # Basic cleaning
    df["title_fr"] = df["title_fr"].fillna("").astype(str).str.strip()
    df["description_fr"] = df["description_fr"].fillna("").astype(str).str.strip()

    # Drop events that have no date at all
    df = df.dropna(subset=["first_begin_dt"]).reset_index(drop=True)

    return df


def split_by_time(df: pd.DataFrame, now_utc: datetime | None = None):
    # Use UTC so comparisons are stable.
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    one_year_ago = now_utc - timedelta(days=365)

    # Past year: [one_year_ago, now)
    past_year = df[(df["first_begin_dt"] >= one_year_ago) & (df["first_begin_dt"] < now_utc)].copy()

    # Upcoming: [now, +inf)
    upcoming = df[df["first_begin_dt"] >= now_utc].copy()

    return past_year, upcoming


def build_document_text(row) -> str:
    date_str = ""
    if pd.notna(row["first_begin_dt"]):
        date_str = row["first_begin_dt"].strftime("%Y-%m-%d %H:%M UTC")

    keywords = row.get("keywords_fr") or []
    keywords = [str(k).strip() for k in keywords if k is not None and str(k).strip()]

    parts = [
        f"Titre: {row.get('title_fr','')}",
        f"Description: {row.get('description_fr','')}",
        f"Date: {date_str}",
        f"Lieu: {row.get('location_name','')}",
        f"Adresse: {row.get('location_address','')}",
        f"Ville: {row.get('location_city','')}",
        f"Mots-clés: {', '.join(keywords)}",
    ]
    return "\n".join([p for p in parts if p and not p.endswith(": ")])


def export(df: pd.DataFrame, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSONL (pratique pour réimporter + garder les listes)
    df.to_json(OUT_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False, date_format="iso")

    # CSV (lisible rapidement, mais les listes seront sérialisées)
    df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8")


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {RAW_PATH}")

    events = load_raw_events(RAW_PATH)
    df = to_dataframe(events)

    # 1) Drop titres vides
    df = df[df["title_fr"].str.len() >= 5].copy()

    # 2) Drop descriptions vraiment vides (optionnel mais conseillé)
    # On garde si description >= 30 chars OU si on a un lieu (sinon trop pauvre)
    df = df[(df["description_fr"].str.len() >= 30) | (df["location_name"].fillna("").str.len() > 0)].copy()

    past_year, upcoming = split_by_time(df)
    df_index = pd.concat([past_year, upcoming], ignore_index=True)

    # 3) Dédoublonnage par uid
    df_index = df_index.drop_duplicates(subset=["uid"]).reset_index(drop=True)

    # 4) Construction du document (AVANT schema final)
    df_index["document"] = df_index.apply(build_document_text, axis=1)

    # 5) Vérifications / coercions de types (AVANT FINAL_COLS)
    df_index["uid"] = df_index["uid"].astype(str)
    df_index["location_lat"] = pd.to_numeric(df_index["location_lat"], errors="coerce")
    df_index["location_lon"] = pd.to_numeric(df_index["location_lon"], errors="coerce")

    # 6) Schéma stable pour une sortie clean
    FINAL_COLS = [
        "uid", "origin_url",
        "title_fr", "description_fr",
        "keywords_fr", "thematique", "type_devenement",
        "first_begin_dt", "first_end_dt", "last_begin_dt", "last_end_dt",
        "location_name", "location_address", "location_city", "location_postal",
        "location_lat", "location_lon",
        "document",
    ]
    df_index = df_index[[c for c in FINAL_COLS if c in df_index.columns]].copy()

    export(df_index, "events_index_ready")

    # versions cohérentes dérivées de df_index (post-nettoyage)
    now_utc = datetime.now(timezone.utc)
    past_year_clean = df_index[df_index["first_begin_dt"] < now_utc].copy()
    upcoming_clean = df_index[df_index["first_begin_dt"] >= now_utc].copy()

    export(past_year_clean, "events_past_year")
    export(upcoming_clean, "events_upcoming")


    print(f"Raw events loaded: {len(events)}")
    print(f"Clean events with dates: {len(df_index)}")
    print(f"Past year events: {len(past_year)}")
    print(f"Upcoming events: {len(upcoming)}")
    print(f"Exports written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
