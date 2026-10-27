"""Push-to-talk in the two voice rooms (client-side, driven by Node).

The rooms' inline scripts are run in a Node ``vm`` sandbox with stubbed browser
globals (``tests/client/ptt_harness.js``) and the real key/pointer handlers are
fired: hold Space → one recording starts; release → the recorder stops and the
utterance is committed (streaming) or uploaded (batch); a sub-200 ms tap sends
nothing; Space while typing is ignored; window blur releases the mic; a dropped
STT socket falls back to batch and reconnects. Regression guard for the VAD
mid-sentence cutoffs that push-to-talk replaced (2026-09-05 prod run).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "backend" / "static"
HARNESS = ROOT / "tests" / "client" / "ptt_harness.js"

_SCRIPT_RE = re.compile(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", re.S)


def _inline_js(page: str, tmp_path: Path) -> Path:
    html = (STATIC / page).read_text(encoding="utf-8")
    out = tmp_path / page.replace(".html", ".js")
    out.write_text("\n;\n".join(_SCRIPT_RE.findall(html)), encoding="utf-8")
    return out


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("page, mode", [
    ("voice_room_functional.html", "functional"),
    ("voice_room.html", "cool"),
])
def test_push_to_talk_handlers(page: str, mode: str, tmp_path: Path) -> None:
    js = _inline_js(page, tmp_path)
    proc = subprocess.run(
        ["node", str(HARNESS), str(js), mode],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL PASS" in proc.stdout
