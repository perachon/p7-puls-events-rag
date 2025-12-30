import sys
import requests

BASE_URL = "http://127.0.0.1:8000"


def _pretty_print_json(obj):
    import json
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _assert_status(r, expected=200):
    if r.status_code != expected:
        print("❌ Unexpected status:", r.status_code)
        print("Response text:", r.text)
        raise AssertionError(f"Expected {expected}, got {r.status_code}")


def test_health():
    print("\n== Test: GET / ==")
    r = requests.get(f"{BASE_URL}/", timeout=10)
    _assert_status(r, 200)
    _pretty_print_json(r.json())
    print("✅ OK")


def test_ask_basic():
    print("\n== Test: POST /ask (basic) ==")
    payload = {"question": "Quels événements sont prévus à Orsay ?"}
    r = requests.post(f"{BASE_URL}/ask", json=payload, timeout=60)
    _assert_status(r, 200)
    data = r.json()
    _pretty_print_json(data)

    # Assertions simples
    assert "answer" in data and isinstance(data["answer"], str)
    assert "sources" in data and isinstance(data["sources"], list)
    print("✅ OK")


def test_ask_with_filters():
    print("\n== Test: POST /ask (filters) ==")
    payload = {
        "question": "Quels événements sont prévus à Orsay ?",
        "allowed_cities": ["Orsay"],
        "future_only": True,
    }
    r = requests.post(f"{BASE_URL}/ask", json=payload, timeout=60)
    _assert_status(r, 200)
    data = r.json()
    _pretty_print_json(data)

    assert "answer" in data and isinstance(data["answer"], str)
    assert "sources" in data and isinstance(data["sources"], list)
    print("✅ OK")


def test_rebuild():
    print("\n== Test: POST /rebuild ==")
    r = requests.post(f"{BASE_URL}/rebuild", timeout=600)
    _assert_status(r, 200)
    data = r.json()
    _pretty_print_json(data)

    assert data.get("status") in ("ok", "error")
    assert "duration_s" in data
    assert "details" in data
    print("✅ OK")


if __name__ == "__main__":
    try:
        test_health()
        test_ask_basic()
        test_ask_with_filters()

        # ⚠️ Rebuild est plus long (et peut être coûteux).
        # test_rebuild()

    except Exception as e:
        print("\n❌ TESTS FAILED:", e)
        sys.exit(1)

    print("\n✅ ALL TESTS PASSED")
    sys.exit(0)
