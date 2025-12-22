from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


PROCESSED_DIR = Path("data/processed")
INDEX_READY = PROCESSED_DIR / "events_index_ready.jsonl"


def test_index_ready_file_exists():
    assert INDEX_READY.exists(), f"Missing file: {INDEX_READY}"


def test_index_ready_loads_and_not_empty():
    df = pd.read_json(INDEX_READY, lines=True)
    assert len(df) > 0, "events_index_ready is empty"


def test_required_columns_present():
    df = pd.read_json(INDEX_READY, lines=True)
    required = {
        "uid",
        "title_fr",
        "first_begin_dt",
        "document",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_missing_dates():
    df = pd.read_json(INDEX_READY, lines=True)
    assert df["first_begin_dt"].notna().all(), "Some events have missing first_begin_dt"


def test_document_not_empty_for_most_rows():
    df = pd.read_json(INDEX_READY, lines=True)
    # On accepte qu'une petite minorité soit pauvre, mais pas la majorité
    non_empty_ratio = (df["document"].fillna("").str.strip().str.len() > 0).mean()
    assert non_empty_ratio >= 0.98, f"Too many empty documents: ratio={non_empty_ratio:.3f}"


def test_time_window_is_within_one_year_or_upcoming():
    """
    Option B: events_index_ready contient:
    - events du dernier an (>= now-365j et < now)
    - events à venir (>= now)
    """
    df = pd.read_json(INDEX_READY, lines=True)

    # Pandas charge souvent les dates en tz-aware déjà; sinon on force utc
    dts = pd.to_datetime(df["first_begin_dt"], utc=True, errors="coerce")
    assert dts.notna().all(), "Some first_begin_dt could not be parsed"

    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)

    ok = (dts >= one_year_ago)  # autorise passé récent + futur
    assert ok.all(), "Found events older than 1 year in events_index_ready"
