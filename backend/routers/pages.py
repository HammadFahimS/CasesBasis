"""Static HTML page routes and small asset serves (login, home, analytics,
simulator, voice room, simulated-case CRUD, ECG-algorithm review) plus the
chrome/drawer injection helpers.

Extracted from ``web_app`` (F3). Every page is read from ``backend/static/`` —
no inline markup lives here. ``web_app`` mounts ``router`` and re-exports the
handlers + ``_serve_with_chrome`` / ``_inject_drawer`` so existing call sites and
tests keep working.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

router = APIRouter()

_STATIC = Path(__file__).parent.parent / "static"


def _serve_with_chrome(filename: str) -> str:
    """Inject the shared top-bar + drawer markup into a page.

    Pages that opt into the new chrome contain a literal `<!--CHROME-->`
    placeholder right after `<body>`; this helper swaps it for the contents
    of `_chrome.html` at serve time so we don't duplicate ~150 lines of
    markup across pages. The `<!--DRAWER-->` placeholder is also supported
    for pages that have their own top bar but still need the drawer + burger
    behaviour (simulator, ECG-algorithm).
    """
    body = (_STATIC / filename).read_text()
    body = body.replace("<!--CHROME-->", (_STATIC / "_chrome.html").read_text())
    body = body.replace("<!--DRAWER-->", (_STATIC / "_drawer.html").read_text())
    return body


def _inject_drawer(html: str) -> str:
    """Swap a `<!--DRAWER-->` placeholder for the drawer-only partial."""
    drawer = (_STATIC / "_drawer.html").read_text()
    return html.replace("<!--DRAWER-->", drawer)


@router.get("/body-image.png")
def body_image() -> FileResponse:
    img_path = _STATIC.parent.parent / "generatedimageofbody.png"
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/static/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(str(_STATIC / "favicon.svg"), media_type="image/svg+xml")


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "login.html").read_text()


@router.get("/simulator", response_class=HTMLResponse)
def simulator_page() -> str:
    return _inject_drawer((_STATIC / "simulator.html").read_text())


@router.get("/simulator/simulated", response_class=HTMLResponse)
def simulated_intro_page() -> str:
    """Intro / explainer page for the new simulated-dynamic-case mode.

    Reached from the 'New case' chooser modal when the resident picks the
    simulated option; the 'Start' button hands off to the simulator with
    `?mode=simulated`. Standalone interstitial — it carries its own slim top
    bar rather than the full home/analytics chrome."""
    return (_STATIC / "simulated_intro.html").read_text()


@router.get("/simulator/voice-room", response_class=HTMLResponse)
def voice_room_page() -> str:
    """Full-screen voice-driven simulation room (the simulated-case runtime UI).

    Standalone immersive page; the case id is read from ?case= in JS and the
    patient HUD + vitals are seeded from the loaded simulated case."""
    return (_STATIC / "voice_room.html").read_text()


@router.get("/crud-simulations", response_class=HTMLResponse)
def crud_simulations_page() -> str:
    """CRUD catalogue for authored simulated dynamic cases."""
    return _serve_with_chrome("crud_simulations.html")


@router.get("/crud-simulations/edit", response_class=HTMLResponse)
def crud_simulations_editor_page() -> str:
    """Graph editor for one simulated case (case id read from ?id= in JS).

    Standalone full-screen tool — carries its own slim top bar instead of the
    shared chrome so the graph canvas gets the full viewport."""
    return (_STATIC / "sim_case_editor.html").read_text()


@router.get("/home", response_class=HTMLResponse)
def home_page() -> str:
    """The new calm landing page after login — greeting, focus card, glance, recs."""
    return _serve_with_chrome("home.html")


@router.get("/analytics", response_class=HTMLResponse)
def analytics_page() -> str:
    """The dense analytics & performance view (was /dashboard)."""
    return _serve_with_chrome("analytics.html")


@router.get("/dashboard")
def dashboard_redirect() -> RedirectResponse:
    """Legacy `/dashboard` URL — kept as a redirect so old bookmarks still land
    somewhere useful. The post-login default is now `/home`."""
    return RedirectResponse(url="/home", status_code=302)


@router.get("/ecg-algorithm", response_class=HTMLResponse)
def ecg_algorithm_page() -> str:
    """Doctor-facing review page for the synthetic ECG generation algorithm."""
    return _serve_with_chrome("ecg_algorithm.html")
