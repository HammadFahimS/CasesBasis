"""Audio hardening in the two voice rooms (client-side, driven by Node).

The 2026-09-05 prod runs had replies on screen that were never heard, with no
server-side signal. The rooms' inline scripts are run in a Node ``vm`` sandbox
(``tests/client/audio_harness.js``) and pushed through those failure modes: a UI
update that throws after the reply arrives, a playback that stalls, a live cue
landing while a turn is in flight, an AudioContext that will not run, and a page
error. In every case the reply is still requested for speech, and every incident
reaches ``/api/voice/client-log``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_push_to_talk import _inline_js

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tests" / "client" / "audio_harness.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("page, mode", [
    ("voice_room_functional.html", "functional"),
    ("voice_room.html", "cool"),
])
def test_replies_are_heard_and_incidents_logged(page: str, mode: str, tmp_path: Path) -> None:
    js = _inline_js(page, tmp_path)
    proc = subprocess.run(
        ["node", str(HARNESS), str(js), mode],
        capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL PASS" in proc.stdout
