from __future__ import annotations

import random

import numpy as np

from schemas.ecg import ECGParameters, LeadFindings

FS = 500  # samples per second
STRIP_DURATION = 2.5  # seconds per lead panel

LEAD_LAYOUT: list[list[str]] = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def _gauss(t: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - mu) / max(sigma, 1e-6)) ** 2)


def _beat_template(
    duration: float,
    findings: LeadFindings,
    pr_s: float,
    qrs_s: float,
    qt_s: float,
) -> np.ndarray:
    """One PQRST complex as a numpy array of `duration` seconds at FS samples/sec."""
    n = int(duration * FS)
    t = np.linspace(0, duration, n)
    wave = np.zeros(n)
    sc = findings.amplitude_scale

    p_mu = pr_s * 0.50
    q_mu = pr_s
    r_mu = pr_s + qrs_s * 0.35
    s_mu = pr_s + qrs_s * 0.72
    t_mu = pr_s + qt_s * 0.65

    # P wave
    if not findings.p_wave_absent:
        wave += _gauss(t, p_mu, pr_s * 0.12, 0.15 * sc)

    # QRS
    wide = findings.qrs_morphology in ("lbbb", "rbbb", "wide")
    r_sig = qrs_s * 0.10 if wide else qrs_s * 0.06
    s_sig = qrs_s * 0.08 if wide else qrs_s * 0.055
    q_amp = -0.25 * sc if findings.pathological_q else -0.08 * sc

    wave += _gauss(t, q_mu, qrs_s * 0.06, q_amp)
    wave += _gauss(t, r_mu, r_sig, 0.90 * sc)
    wave += _gauss(t, s_mu, s_sig, -0.15 * sc)

    # LBBB notched R: add a second R shoulder
    if findings.qrs_morphology == "lbbb":
        wave += _gauss(t, r_mu + qrs_s * 0.3, r_sig * 0.8, 0.40 * sc)

    # RBBB: RSR' pattern – extra positive deflection after S
    if findings.qrs_morphology == "rbbb":
        wave += _gauss(t, s_mu + qrs_s * 0.35, r_sig * 0.7, 0.45 * sc)

    # Delta wave (WPW): slurred upstroke before R
    if findings.qrs_morphology == "delta":
        wave += _gauss(t, q_mu - qrs_s * 0.15, qrs_s * 0.12, 0.30 * sc)

    # ST segment offset in the iso-electric region between QRS end and T peak
    st_mv = findings.st_elevation_mv
    if abs(st_mv) > 0.005:
        st_start = s_mu + qrs_s * 0.08
        st_end = t_mu - (qt_s - qrs_s) * 0.25
        mask = (t >= st_start) & (t <= st_end)
        wave[mask] += st_mv * 0.30

    # T wave
    t_amp = -0.20 * sc if findings.t_inverted else (0.35 * sc + st_mv * 0.06)
    wave += _gauss(t, t_mu, (qt_s - qrs_s) * 0.18, t_amp)

    return wave


def _build_signal(
    params: ECGParameters,
    findings: LeadFindings,
    duration: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(duration * FS)
    t_axis = np.linspace(0, duration, n)
    signal = np.zeros(n)
    rng = random.Random(seed)

    pr_s = params.pr_interval_ms / 1000.0
    qrs_s = params.qrs_duration_ms / 1000.0
    qt_s = params.qtc_ms / 1000.0
    rhythm = params.rhythm
    beat_interval = 60.0 / params.heart_rate

    # ── VF: chaotic noise ───────────────────────────────────────────────────────
    if rhythm == "vf":
        rng2 = np.random.default_rng(seed)
        signal = rng2.normal(0, 0.18, n)
        signal += 0.10 * np.sin(2 * np.pi * 5.5 * t_axis)
        signal += 0.06 * np.sin(2 * np.pi * 11 * t_axis)
        return t_axis, signal

    # ── Flutter: sawtooth baseline + QRS ────────────────────────────────────────
    if rhythm == "flutter":
        flutter_interval = 0.20  # 300 bpm
        t_fl = 0.0
        while t_fl < duration:
            idx = int(t_fl * FS)
            n_fl = int(flutter_interval * FS)
            t_loc = np.linspace(0, flutter_interval, n_fl)
            sawtooth = -0.09 * (t_loc / flutter_interval)
            sawtooth += _gauss(t_loc, flutter_interval * 0.88, 0.012, 0.09)
            end = min(idx + n_fl, n)
            signal[idx:end] += sawtooth[: end - idx]
            t_fl += flutter_interval
        qrs_findings = LeadFindings(
            p_wave_absent=True,
            st_elevation_mv=findings.st_elevation_mv,
            t_inverted=findings.t_inverted,
            qrs_morphology=findings.qrs_morphology,
            amplitude_scale=findings.amplitude_scale,
        )
        t_qrs = 0.0
        while t_qrs < duration:
            idx = int(t_qrs * FS)
            tpl = _beat_template(beat_interval, qrs_findings, pr_s=0.04, qrs_s=qrs_s, qt_s=qt_s)
            end = min(idx + len(tpl), n)
            signal[idx:end] += tpl[: end - idx]
            t_qrs += beat_interval
        return t_axis, signal

    # ── Complete heart block: independent P waves + escape QRS ──────────────────
    if rhythm == "complete_heart_block":
        p_interval = 60.0 / 75.0  # atrial rate ~75 bpm
        p_findings = LeadFindings(
            p_wave_absent=False,
            amplitude_scale=findings.amplitude_scale * 0.8,
        )
        t_p = 0.0
        while t_p < duration:
            idx = int(t_p * FS)
            n_p = int(p_interval * 0.45 * FS)
            t_loc = np.linspace(0, p_interval * 0.45, n_p)
            p_mu = p_interval * 0.20
            p_wave = _gauss(t_loc, p_mu, p_interval * 0.05, 0.15 * p_findings.amplitude_scale)
            end = min(idx + n_p, n)
            signal[idx:end] += p_wave[: end - idx]
            t_p += p_interval
        escape_findings = LeadFindings(
            p_wave_absent=True,
            st_elevation_mv=findings.st_elevation_mv,
            t_inverted=findings.t_inverted,
            pathological_q=findings.pathological_q,
            qrs_morphology=findings.qrs_morphology,
            amplitude_scale=findings.amplitude_scale,
        )
        t_qrs = beat_interval * 0.4  # slight offset so P and QRS are dissociated
        while t_qrs < duration:
            idx = int(t_qrs * FS)
            tpl = _beat_template(beat_interval, escape_findings, pr_s=0.06, qrs_s=qrs_s, qt_s=qt_s)
            end = min(idx + len(tpl), n)
            signal[idx:end] += tpl[: end - idx]
            t_qrs += beat_interval
        return t_axis, signal

    # ── Standard beat tiling (sinus variants, AF, VT, SVT) ─────────────────────
    force_no_p = rhythm in ("af", "vt", "svt") or findings.p_wave_absent
    eff = LeadFindings(
        st_elevation_mv=findings.st_elevation_mv,
        t_inverted=findings.t_inverted,
        pathological_q=findings.pathological_q,
        qrs_morphology=findings.qrs_morphology,
        amplitude_scale=findings.amplitude_scale,
        p_wave_absent=force_no_p,
    )
    t_beat = 0.0
    while t_beat < duration:
        jitter = rng.uniform(-0.15, 0.15) * beat_interval if rhythm == "af" else 0.0
        interval = max(beat_interval + jitter, 0.20)
        idx = int(t_beat * FS)
        tpl = _beat_template(interval, eff, pr_s, qrs_s, qt_s)
        end = min(idx + len(tpl), n)
        signal[idx:end] += tpl[: end - idx]
        t_beat += interval

    return t_axis, signal


def synthesize_ecg_svg(params: ECGParameters) -> str:
    """Return a 12-lead ECG as a compact inline-embeddable SVG string."""
    COLS = 4
    ROWS = 3
    PW = 218   # panel width px
    PH = 130   # panel height px
    PAD_L = 6  # left pad inside panel (after label)
    PAD_T = 16 # top pad for label
    PAD_B = 4  # bottom pad
    PLOT_W = PW - PAD_L - 4
    PLOT_H = PH - PAD_T - PAD_B
    # Amplitude range: -0.8 to 1.6 mV → 2.4 mV total
    AMP_RANGE = 2.4
    Y_MID_MV = 0.4  # where 0 mV sits (slightly above mid)
    Y_SCALE = PLOT_H / AMP_RANGE
    X_SCALE = PLOT_W / STRIP_DURATION

    # Lead II rhythm strip at the bottom: same panel height, full SVG width,
    # 10 s of continuous signal so the reader can assess rhythm across many
    # beats. Uses its own X scale (smaller) so 10 s fits, which is why we
    # need a separate grid pattern below.
    RHYTHM_DURATION = 10.0
    RHYTHM_LEAD = "II"
    total_w = COLS * PW
    rhythm_plot_w = total_w - PAD_L - 4
    X_SCALE_RH = rhythm_plot_w / RHYTHM_DURATION

    total_h = ROWS * PH + PH  # 3 grid rows + 1 rhythm strip row

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
        f'style="width:100%;height:auto;display:block;background:#fff;font-family:Arial,Helvetica,sans-serif">',
    ]

    # Shared defs for minor / major grid patterns
    out.append("<defs>")
    out.append(
        f'<pattern id="mingrid" x="0" y="0" width="{X_SCALE*0.04:.2f}" height="{Y_SCALE*0.1:.2f}" patternUnits="userSpaceOnUse">'
        f'<path d="M {X_SCALE*0.04:.2f} 0 L 0 0 0 {Y_SCALE*0.1:.2f}" fill="none" stroke="#ffdddd" stroke-width="0.3"/>'
        f"</pattern>"
    )
    out.append(
        f'<pattern id="majgrid" x="0" y="0" width="{X_SCALE*0.20:.2f}" height="{Y_SCALE*0.5:.2f}" patternUnits="userSpaceOnUse">'
        f'<path d="M {X_SCALE*0.20:.2f} 0 L 0 0 0 {Y_SCALE*0.5:.2f}" fill="none" stroke="#ffaaaa" stroke-width="0.6"/>'
        f"</pattern>"
    )
    # Rhythm strip uses its own grid pattern with the wider X scale so the
    # 0.04 s / 0.20 s gridlines still mean the same thing in time, just
    # spaced differently in pixels.
    out.append(
        f'<pattern id="mingrid_rh" x="0" y="0" width="{X_SCALE_RH*0.04:.2f}" height="{Y_SCALE*0.1:.2f}" patternUnits="userSpaceOnUse">'
        f'<path d="M {X_SCALE_RH*0.04:.2f} 0 L 0 0 0 {Y_SCALE*0.1:.2f}" fill="none" stroke="#ffdddd" stroke-width="0.3"/>'
        f"</pattern>"
    )
    out.append(
        f'<pattern id="majgrid_rh" x="0" y="0" width="{X_SCALE_RH*0.20:.2f}" height="{Y_SCALE*0.5:.2f}" patternUnits="userSpaceOnUse">'
        f'<path d="M {X_SCALE_RH*0.20:.2f} 0 L 0 0 0 {Y_SCALE*0.5:.2f}" fill="none" stroke="#ffaaaa" stroke-width="0.6"/>'
        f"</pattern>"
    )
    out.append("</defs>")

    step = max(1, FS // 100)  # plot at ~100 Hz for compact paths

    for row_i, row_leads in enumerate(LEAD_LAYOUT):
        for col_i, lead_name in enumerate(row_leads):
            ox = col_i * PW
            oy = row_i * PH
            plot_x = ox + PAD_L
            plot_y = oy + PAD_T
            clip_id = f"c{row_i}{col_i}"

            # Panel background + border
            out.append(
                f'<rect x="{ox}" y="{oy}" width="{PW}" height="{PH}" '
                f'fill="#fff9f9" stroke="#e8e8e8" stroke-width="0.5"/>'
            )

            # Clip path for waveform
            out.append(
                f'<clipPath id="{clip_id}">'
                f'<rect x="{plot_x}" y="{plot_y}" width="{PLOT_W}" height="{PLOT_H}"/>'
                f"</clipPath>"
            )

            # Grid (minor then major, both as pattern fills)
            out.append(
                f'<rect x="{plot_x}" y="{plot_y}" width="{PLOT_W}" height="{PLOT_H}" fill="url(#mingrid)"/>'
            )
            out.append(
                f'<rect x="{plot_x}" y="{plot_y}" width="{PLOT_W}" height="{PLOT_H}" fill="url(#majgrid)"/>'
            )

            # Isoelectric baseline (0 mV)
            y_zero = plot_y + (AMP_RANGE - Y_MID_MV) / AMP_RANGE * PLOT_H
            out.append(
                f'<line x1="{plot_x}" y1="{y_zero:.1f}" x2="{plot_x+PLOT_W}" y2="{y_zero:.1f}" '
                f'stroke="#ffbbbb" stroke-width="0.8"/>'
            )

            # Lead label
            out.append(
                f'<text x="{ox+3}" y="{oy+11}" font-size="10" font-weight="bold" fill="#444">{lead_name}</text>'
            )

            # Build waveform polyline
            findings = params.leads.get(lead_name) or LeadFindings()
            t_axis, signal = _build_signal(
                params, findings, STRIP_DURATION, seed=abs(hash(lead_name)) % 9999
            )
            pts: list[str] = []
            for t_val, sig_val in zip(t_axis[::step], signal[::step]):
                px = plot_x + t_val * X_SCALE
                # y: higher mV → lower pixel y
                py = plot_y + (AMP_RANGE - sig_val - Y_MID_MV) / AMP_RANGE * PLOT_H
                py = max(plot_y, min(plot_y + PLOT_H, py))  # clamp
                pts.append(f"{px:.1f},{py:.1f}")

            out.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="#111" '
                f'stroke-width="1.2" stroke-linejoin="round" clip-path="url(#{clip_id})"/>'
            )

    # ── Lead II rhythm strip (row 4, full width, 10 s) ──────────────────────
    rh_ox = 0
    rh_oy = ROWS * PH
    rh_plot_x = rh_ox + PAD_L
    rh_plot_y = rh_oy + PAD_T
    rh_clip_id = "crh"

    # Panel background + border
    out.append(
        f'<rect x="{rh_ox}" y="{rh_oy}" width="{total_w}" height="{PH}" '
        f'fill="#fff9f9" stroke="#e8e8e8" stroke-width="0.5"/>'
    )
    # Clip path for waveform
    out.append(
        f'<clipPath id="{rh_clip_id}">'
        f'<rect x="{rh_plot_x}" y="{rh_plot_y}" width="{rhythm_plot_w}" height="{PLOT_H}"/>'
        f"</clipPath>"
    )
    # Grid (wider-scale minor + major)
    out.append(
        f'<rect x="{rh_plot_x}" y="{rh_plot_y}" width="{rhythm_plot_w}" height="{PLOT_H}" fill="url(#mingrid_rh)"/>'
    )
    out.append(
        f'<rect x="{rh_plot_x}" y="{rh_plot_y}" width="{rhythm_plot_w}" height="{PLOT_H}" fill="url(#majgrid_rh)"/>'
    )
    # Iso-electric baseline (0 mV)
    y_zero_rh = rh_plot_y + (AMP_RANGE - Y_MID_MV) / AMP_RANGE * PLOT_H
    out.append(
        f'<line x1="{rh_plot_x}" y1="{y_zero_rh:.1f}" x2="{rh_plot_x + rhythm_plot_w}" y2="{y_zero_rh:.1f}" '
        f'stroke="#ffbbbb" stroke-width="0.8"/>'
    )
    # Lead label
    out.append(
        f'<text x="{rh_ox + 3}" y="{rh_oy + 11}" font-size="10" font-weight="bold" fill="#444">{RHYTHM_LEAD}</text>'
    )
    # Build the 10-second Lead II signal using the same parameters as the grid
    rh_findings = params.leads.get(RHYTHM_LEAD) or LeadFindings()
    t_axis_rh, signal_rh = _build_signal(
        params, rh_findings, RHYTHM_DURATION, seed=abs(hash(RHYTHM_LEAD)) % 9999
    )
    pts_rh: list[str] = []
    for t_val, sig_val in zip(t_axis_rh[::step], signal_rh[::step]):
        px = rh_plot_x + t_val * X_SCALE_RH
        py = rh_plot_y + (AMP_RANGE - sig_val - Y_MID_MV) / AMP_RANGE * PLOT_H
        py = max(rh_plot_y, min(rh_plot_y + PLOT_H, py))
        pts_rh.append(f"{px:.1f},{py:.1f}")
    out.append(
        f'<polyline points="{" ".join(pts_rh)}" fill="none" stroke="#111" '
        f'stroke-width="1.2" stroke-linejoin="round" clip-path="url(#{rh_clip_id})"/>'
    )

    out.append("</svg>")
    return "\n".join(out)
