"""The dependency rule, enforced on the import graph rather than on text.

The first version searched each file's text for "app.intelligence" and for
lines starting "import google", so a relative import (`from ..intelligence
import agent`), `importlib.import_module("google.genai")`, `import os,
google.genai`, or knowledge importing tools all got past it. This one parses
every module, resolves relative imports, counts importlib and __import__
calls (a name it can't read counts against the file), and checks each edge
against the layer table in the README."""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"

# Which app packages each layer may import, as in the README's layer table.
# config and observability are shared by everyone; api is the HTTP edge the
# client talks to; main is the composition root and may import anything.
SHARED = {"config", "observability"}
ALLOWED = {
    "api": SHARED | {"intelligence", "knowledge", "inference", "tools"},
    "intelligence": SHARED | {"inference", "knowledge", "tools"},
    "tools": SHARED | {"knowledge", "inference"},
    "knowledge": SHARED | {"inference"},
    "inference": SHARED,
    "config": set(),
    "observability": {"config"},
}
VENDOR = "google"  # only the inference layer may import the Gemini SDK


def _layer(path: Path, app: Path) -> str:
    rel = path.relative_to(app)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def imports(path: Path, app: Path) -> set[str]:
    """Every module `path` imports, as absolute dotted names; "?" for an
    importlib/__import__ call whose module isn't a string literal."""
    package = (app.name, *path.relative_to(app).parent.parts)
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else ()
            module = ".".join((*base, node.module) if node.module else base)
            found.add(module)
            # `from app import tools` imports a package, not a name
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("import_module", "__import__"):
                arg = node.args[0] if node.args else None
                found.add(arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else "?")
    return found


def violations(app: Path = APP) -> list[str]:
    problems = []
    for path in sorted(app.rglob("*.py")):
        layer = _layer(path, app)
        if layer in ("main", "__init__"):
            continue
        for module in imports(path, app):
            where = path.relative_to(app.parent)
            if module == "?":
                problems.append(f"{where}: an import whose module can't be read")
            elif (module == VENDOR or module.startswith(VENDOR + ".")) and layer != "inference":
                problems.append(f"{where}: imports the vendor SDK ({module})")
            elif module.startswith(app.name + "."):
                target = module.split(".")[1]
                if target != layer and target not in ALLOWED.get(layer, set()):
                    problems.append(f"{where}: {layer} imports {target} ({module})")
    return problems


def test_every_import_follows_the_layer_rules():
    assert violations() == []


def test_every_layer_has_a_rule():
    layers = {_layer(p, APP) for p in APP.rglob("*.py")} - {"main", "__init__"}
    assert layers <= set(ALLOWED)


def test_the_bypasses_that_fooled_the_text_search_are_caught(tmp_path):
    app = tmp_path / "app"
    files = {
        "knowledge/a.py": "from ..intelligence.agent import Agent\n",
        "knowledge/b.py": "import importlib\nsdk = importlib.import_module('google.genai')\n",
        "tools/c.py": "import os, google.genai\n",
        "knowledge/d.py": "from app import tools\n",
        "knowledge/e.py": "import importlib\nname = 'app.' + 'intelligence'\nimportlib.import_module(name)\n",
        "inference/ok.py": "from google import genai\nfrom app.config import get_settings\n",
        "knowledge/ok.py": "from ..inference.provider import LLMProvider\n",
    }
    for name, code in files.items():
        (app / name).parent.mkdir(parents=True, exist_ok=True)
        (app / name).write_text(code)

    found = "\n".join(violations(app))

    assert "knowledge/a.py: knowledge imports intelligence" in found
    assert "knowledge/b.py: imports the vendor SDK" in found
    assert "tools/c.py: imports the vendor SDK" in found
    assert "knowledge/d.py: knowledge imports tools" in found
    assert "knowledge/e.py: an import whose module can't be read" in found
    assert "ok.py" not in found
