"""The dependency rule, enforced: only app.inference may import the vendor SDK."""

import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
VENDOR = re.compile(r"^\s*(from|import)\s+google(\.|\s)", re.M)


def test_only_inference_imports_vendor_sdk():
    offenders = [
        p.relative_to(APP)
        for p in APP.rglob("*.py")
        if VENDOR.search(p.read_text()) and p.parent.name != "inference"
    ]
    assert not offenders, f"vendor SDK imported outside inference layer: {offenders}"


def test_layers_do_not_import_upward():
    # knowledge and tools must never import intelligence or api
    for layer in ("inference", "knowledge", "tools"):
        for p in (APP / layer).rglob("*.py"):
            src = p.read_text()
            assert "app.intelligence" not in src, f"{p} imports intelligence"
            assert "app.api" not in src, f"{p} imports api"
    # inference must not import knowledge or tools
    for p in (APP / "inference").rglob("*.py"):
        src = p.read_text()
        assert "app.knowledge" not in src and "app.tools" not in src, p
