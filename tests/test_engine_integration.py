"""Integration-level tests — validates data loading and model consistency."""

import json
from pathlib import Path

from src.matching.engine import DATA_DIR
from src.matching.models import AttendeeProfile


def test_sample_profiles_load():
    path = DATA_DIR / "sample_profiles.json"
    assert path.exists(), f"Missing {path}"
    with open(path) as f:
        raw = json.load(f)
    assert len(raw) == 5
    profiles = [AttendeeProfile(**p) for p in raw]
    assert all(p.name for p in profiles)
    assert all(p.id for p in profiles)


def test_minimal_profiles_load():
    path = DATA_DIR / "sample_profiles_minimal.json"
    assert path.exists(), f"Missing {path}"
    with open(path) as f:
        raw = json.load(f)
    profiles = [AttendeeProfile(**p) for p in raw]
    assert len(profiles) == 5
    for p in profiles:
        assert p.stated_goal or p.looking_for


def test_minimal_profile_has_product():
    path = DATA_DIR / "sample_profiles_minimal.json"
    with open(path) as f:
        raw = json.load(f)
    for entry in raw:
        assert "product" in entry or "looking_for" in entry
    profiles = [AttendeeProfile(**p) for p in raw]
    for p in profiles:
        assert p.product is not None
