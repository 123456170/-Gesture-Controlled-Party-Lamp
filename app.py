import os
import io
import wave
import time
import math
import random

import numpy as np
import streamlit as st
import cv2

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

st.set_page_config(
    page_title="Gesture Party Lamp",
    page_icon="🪩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FRAME_DT = 1.0 / 24.0
TRACK_OPTIONS = ["Steady party beat", "Ambient texture"]

DEFAULTS = {
    "auto_demo": True,
    "use_camera": False,
    "manual_height": 0.15,
    "hand_height": 0.15,
    "manual_raise": False,
    "auto_raise": False,
    "camera_raise": False,
    "pending_swipe": False,
    "chill_hold_start": None,
    "chill_target": 0.0,
    "chill": 0.0,
    "hue": random.random(),
    "pulse": 0.0,
    "flash": 0.0,
    "beat_flash": 0.0,
    "last_beat_index": -1,
    "play_start": time.time(),
    "demo_start": time.time(),
    "last_frame_time": time.time(),
    "caption": "Starting live demo...",
    "last_event": "",
    "auto_phase": 0.0,
    "auto_fired": set(),
    "camera_error": None,
    "cap": None,
    "hands": None,
    "camera_frame": None,
    "cam_hist": [],
    "cam_chill_start": None,
    "last_camera_swipe": 0.0,
    "audio_backend": None,
    "pygame_sound": None,
    "audio_error": None,
    "energy_scale": 1.0,
    "duration": 120.0,
    "beats": np.array([]),
    "energies": np.array([0.0]),
    "energy_dt": 0.05,
    "sr": 22050,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def lerp(a, b, t):
    return a + (b - a) * t


def put_text(img, text, org, scale=0.55, color=(230, 230, 230), thickness=1):
    cv2.putText(
        img,
        str(text),
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def hsv_to_bgr(h, s, v):
    h = float(h) % 1.0
    s = clamp(s)
    v = clamp(v)

    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def make_wav_bytes(pcm, sr):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buffer.getvalue()


def synthesize_audio(kind):
    duration = 120.0
    sr = 22050
    n = int(duration * sr)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(1234)

    def add_event(start, dur, fn):
        s = int(start * sr)
        e = min(n, int((start + dur) * sr))
        if s < 0 or s >= e:
            return
        tt = np.arange(e - s, dtype=np.float32) / float(sr)
        chunk = fn(tt)
        audio[s:e] += chunk[: e - s].astype(np.float32)

    if kind == TRACK_OPTIONS[0]:
        bpm = 122.0
        beat = 60.0 / bpm
        bass_notes = [55.0, 55.0, 65.41, 49.0]

        for i, start in enumerate(np.arange(0.0, duration, beat)):
            def kick_fn(tt):
                freq = 150.0 * np.exp(-tt * 18.0) + 48.0
                phase = 2.0 * np.pi * np.cumsum(freq) / float(sr)
                return np.sin(phase) * np.exp(-tt * 25.0) * 0.95

            add_event(start, 0.24, kick_fn)

            def hat_fn(tt):
                noise = rng.random(len(tt)) * 2.0 - 1.0
                return noise * np.exp(-tt * 95.0) * 0.11

            add_event(start + beat / 2.0, 0.06, hat_fn)

            if i % 2 == 1:
                def clap_fn(tt):
                    noise = rng.random(len(tt)) * 2.0 - 1.0
                    return noise * np.exp(-tt * 40.0) * 0.16

                add_event(start, 0.12, clap_fn)

            note = bass_notes[(i // 4) % len(bass_notes)]

            def bass_fn(tt, note=note):
                env = np.exp(-tt * 2.4)
                fundamental = np.sin(2.0 * np.pi * note * tt)
                harmonic = 0.35 * np.sin(2.0 * np.pi * note * 2.0 * tt)
                return (fundamental + harmonic) * env * 0.15

            add_event(start, beat * 0.92, bass_fn)

        pad = 0.04 * np.sin(2.0 * np.pi * 220.0 * np.arange(n, dtype=np.float32) / sr)
        pad *= 0.35 + 0.65 * np.sin(2.0 * np.pi * 0.05 * np.arange(n, dtype=np.float32) / sr)
        audio += pad.astype(np.float32)

    else:
        t = np.arange(n, dtype=np.float32) / float(sr)

        audio += 0.16 * np.sin(2.0 * np.pi * 110.0 * t) * (
            0.45 + 0.55 * np.sin(2.0 * np.pi * 0.023 * t)
        )
        audio += 0.10 * np.sin(2.0 * np.pi * 164.8 * t) * (
            0.45 + 0.55 * np.sin(2.0 * np.pi * 0.017 * t + 1.3)
        )
        audio += 0.07 * np.sin(2.0 * np.pi * 220.0 * t) * (
            0.45 + 0.55 * np.sin(2.0 * np.pi * 0.011 * t + 2.1)
        )

        rng2 = np.random.default_rng(99)
        start = 2.5

        while start < duration - 5.0:
            dur = 3.0
            s = int(start * sr)
            e = min(n, int((start + dur) * sr))
            if s < e:
                tt = np.arange(e - s, dtype=np.float32) / float(sr)
                freq = float(rng2.choice([220.0, 277.18, 329.63, 392.0]))
                env = np.sin(np.pi * np.linspace(0.0, 1.0, e - s, dtype=np.float32)) ** 2
                audio[s:e] += np.sin(2.0 * np.pi * freq * tt) * env * 0.14

            start += float(rng2.uniform(5.0, 9.0))

    peak = float(np.max(np.abs(audio))) if n else 1.0
    if peak > 1e-6:
        audio = audio / peak * 0.92

    fade = int(sr * 0.4)
    if fade > 0 and len(audio) > 2 * fade:
        audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    return audio.astype(np.float32), sr


@st.cache_data(show_spinner=False)
def get_track_assets(kind):
    audio, sr = synthesize_audio(kind)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    pcm_bytes = pcm.tobytes()
    wav_bytes = make_wav_bytes(pcm, sr)
    duration = float(len(audio)) / float(sr)
    return audio, sr, pcm_bytes, wav_bytes, duration


def detect_beats(audio, sr, sensitivity):
    frame_len = int(sr * 0.05)
    if frame_len <= 0 or len(audio) < frame_len * 4:
        return np.array([]), np.array([0.0]), 0.05

    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1)).astype(np.float32)

    if len(energy) > 3:
        energy = np.convolve(energy, np.ones(3, dtype=np.float32) / 3.0, mode="same")

    window = 20
    if len(energy) > window:
        rolling = np.convolve(energy, np.ones(window, dtype=np.float32) / float(window), mode="same")
    else:
        rolling = np.full_like(energy, float(np.mean(energy)) if len(energy) else 0.001)

    rolling = np.maximum(rolling, 0.0015)

    beats = []
    last_beat_time = -10.0
    min_interval = 0.27

    for i in range(1, len(energy) - 1):
        if (
            energy[i] > rolling[i] * sensitivity
            and energy[i] >= energy[i - 1]
            and energy[i] >= energy[i + 1]
            and energy[i] > 0.004
        ):
            t = i * frame_len / float(sr)
            if t - last_beat_time >= min_interval:
                beats.append(t)
                last_beat_time = t

    return np.array(beats, dtype=np.float32), energy, float(frame_len) / float(sr)


@st.cache_data(show_spinner=False)
def get_beat_data(kind, sensitivity):
    audio, sr, _, _, _ = get_track_assets(kind)
    beats, energies, energy_dt = detect_beats(audio, sr, float(sensitivity))
    return beats, energies, energy_dt, sr


def start_audio(kind):
    _, sr, pcm_bytes, wav_bytes, _ = get_track_assets(kind)
    st.session_state.play_start = time.time()
    st.session_state.audio_error = None

    try:
        import pygame

        init_info = pygame.mixer.get_init()
        if init_info:
            freq, _, channels = init_info
            if freq != sr or channels != 1:
                pygame.mixer.quit()

        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(sr, -16, 1, 512)
            pygame.mixer.init()

        if st.session_state.get("pygame_sound") is not None:
            try:
                st.session_state.pygame_sound.stop()
            except Exception:
                pass

        sound = pygame.mixer.Sound(buffer=pcm_bytes)
        sound.set_volume(0.85)
        sound.play(loops=-1)

        st.session_state.pygame_sound = sound
        st.session_state.audio_backend = "pygame"

    except Exception as exc:
        st.session_state.audio_backend = "fallback"
        st.session_state.audio_error = str(exc)


def bpm_estimate(beats, pos):
    if len(beats) < 4:
        return 0.0

    idx = int(np.searchsorted(beats, pos))
    start = max(0, idx - 10)
    recent = beats[start : min(len(beats), idx + 2)]

    if len(recent) < 4:
        recent = beats[: min(12, len(beats))]

    diffs = np.diff(recent)
    if len(diffs) == 0:
        return 0.0

    diffs = diffs[(diffs > 0.25) & (diffs < 2.5)]
    if len(diffs) == 0:
        return 0.0

    return float(60.0 / np.median(diffs))


def nearest_beat_distance(beats, pos):
    if len(beats) == 0:
        return 10.0

    idx = int(np.searchsorted(beats, pos))
    best = 10.0

    if idx < len(beats):
        best = min(best, abs(float(beats[idx]) - pos))
    if idx > 0:
        best = min(best, abs(pos - float(beats[idx - 1])))

    return best


def apply_auto_demo(now):
    if not st.session_state.get("auto_demo", False):
        st.session_state.auto_raise = False
        return None

    t = (now - st.session_state.demo_start) % 120.0
    prev = st.session_state.get("auto_phase", 0.0)

    if t < prev - 1.0:
        st.session_state.auto_fired = set()
        st.session_state.chill_target = 0.0
        st.session_state.chill_hold_start = None

    fired = st.session_state.auto_fired
    st.session_state.auto_raise = False

    height = 0.12
    caption = "Live demo"

    if t < 15.0:
        caption = "0:00-0:15 Audio-reactive baseline: lamp pulses on detected beats."

    elif t < 40.0:
        p = (t - 15.0) / 25.0
        height = 0.12 + 0.83 * p
        caption = "0:15-0:40 Raising hand speeds up hue cycling."

    elif t < 60.0:
        height = 0.55
        caption = "0:40-1:00 Fast horizontal swipe triggers color jump + flash."

        if 42.0 <= t < 42.4 and "swipe1" not in fired:
            st.session_state.pending_swipe = True
            fired.add("swipe1")
            st.session_state.last_event = "Auto swipe 1"

        if 52.0 <= t < 52.4 and "swipe2" not in fired:
            st.session_state.pending_swipe = True
            fired.add("swipe2")
            st.session_state.last_event = "Auto swipe 2"

    elif t < 85.0:
        height = 0.70
        caption = "1:00-1:25 Raise-the-roof: beat-synced strobe while held."
        st.session_state.auto_raise = 62.0 <= t < 83.0

    elif t < 105.0:
        height = 0.08
        caption = "1:25-1:45 Chill-out: holding both palms ramps down to calm glow."

        if 85.0 <= t < 86.5:
            if st.session_state.chill_hold_start is None:
                st.session_state.chill_hold_start = now
                st.session_state.last_event = "Auto chill hold"
        else:
            st.session_state.chill_target = 1.0

    else:
        height = 0.08
        caption = "1:45-2:00 Calm ending. Hybrid CV + audio-reactive demo."

    if t < 1.0:
        st.session_state.chill_target = 0.0
        st.session_state.chill_hold_start = None

    st.session_state.auto_phase = t
    st.session_state.caption = caption
    return height


def update_camera(now):
    if not st.session_state.get("use_camera", False):
        st.session_state.camera_raise = False
        return None

    if st.session_state.camera_error is not None:
        return None

    try:
        import mediapipe as mp

        if st.session_state.get("cap") is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Webcam could not be opened.")

            hands = mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            st.session_state.cap = cap
            st.session_state.hands = hands
            st.session_state.camera_error = None

        cap = st.session_state.cap
        hands = st.session_state.hands

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("No webcam frame received.")

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        height = None
        swipe = False
        raise_roof = False
        chill_signal = False
        hand_count = 0
        centers = []

        if result.multi_hand_landmarks:
            hand_count = len(result.multi_hand_landmarks)

            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                x = (lm[0].x + lm[9].x) / 2.0
                y = (lm[0].y + lm[9].y) / 2.0
                centers.append((x, y, 1.0 - y))

            if centers:
                height = max(center[2] for center in centers)

                hist = st.session_state.cam_hist
                hist.append((now, centers[0][0], centers[0][1]))

                while hist and now - hist[0][0] > 0.45:
                    hist.pop(0)

                if len(hist) >= 4:
                    t0, x0, y0 = hist[0]
                    t1, x1, y1 = hist[-1]
                    dt_hand = max(0.02, t1 - t0)
                    vx = (x1 - x0) / dt_hand
                    vy = (y1 - y0) / dt_hand

                    if abs(vx) > 1.15 and now - st.session_state.get("last_camera_swipe", 0.0) > 0.8:
                        swipe = True
                        st.session_state.last_camera_swipe = now

                    if (
                        hand_count >= 2
                        and abs(vx) < 0.07
                        and abs(vy) < 0.07
                        and 0.25 < centers[0][2] < 0.80
                    ):
                        chill_signal = True

            if hand_count >= 2 and len(centers) >= 2:
                ys = [center[1] for center in centers[:2]]
                if max(ys) < 0.48:
                    raise_roof = True

                    if len(st.session_state.cam_hist) >= 4:
                        t0, _, y0 = st.session_state.cam_hist[0]
                        t1, _, y1 = st.session_state.cam_hist[-1]
                        vertical_speed = abs(y1 - y0) / max(0.02, t1 - t0)
                        if vertical_speed < 0.015:
                            raise_roof = False

            try:
                mp_drawing = mp.solutions.drawing_utils
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks)
            except Exception:
                pass

        else:
            st.session_state.cam_hist.clear()

        if chill_signal:
            if st.session_state.get("cam_chill_start") is None:
                st.session_state.cam_chill_start = now
            elif now - st.session_state.cam_chill_start >= 1.5:
                st.session_state.chill_target = 1.0
                st.session_state.last_event = "Camera chill-out activated"
        else:
            st.session_state.cam_chill_start = None

        st.session_state.camera_raise = raise_roof

        if swipe:
            st.session_state.pending_swipe = True
            st.session_state.last_event = "Camera swipe"

        put_text(frame, f"Hands: {hand_count}", (10, 25), 0.6, (255, 255, 255), 1)
        st.session_state.camera_frame = cv2.resize(frame, (360, 202))

        return {"height": height}

    except Exception as exc:
        st.session_state.camera_error = str(exc)
        st.session_state.camera_raise = False
        return None


def render_frame(s):
    width, height = 760, 430
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (16, 12, 22)

    cv2.rectangle(frame, (0, height - 70), (width, height), (24, 18, 32), -1)

    center = (width // 2, height // 2 + 10)
    base_radius = 82

    pulse = s["pulse"] * (1.0 - s["chill"] * 0.8)
    flash = s["flash"] * (1.0 - s["chill"] * 0.7)

    radius = int(base_radius * (1.0 + 0.35 * pulse + 0.12 * flash))

    brightness = 1.0
    if s["strobe_active"]:
        brightness = 1.0 if s["strobe_on"] else 0.06

    brightness += flash * 0.5
    brightness *= 1.0 - 0.4 * s["chill"]

    hue = s["hue"]
    sat = s["sat"]
    val = clamp(s["val"] * brightness, 0.0, 1.0)
    color = hsv_to_bgr(hue, sat, val)

    glow_layers = [
        (2.35, 0.10),
        (1.95, 0.16),
        (1.55, 0.24),
        (1.25, 0.38),
    ]

    for mult, alpha in glow_layers:
        glow_color = tuple(
            int(channel * alpha * (0.35 + 0.65 * clamp(brightness)))
            for channel in color
        )
        cv2.circle(
            frame,
            center,
            max(1, int(radius * mult)),
            glow_color,
            -1,
            cv2.LINE_AA,
        )

    core_val = clamp(val * 1.25 + flash * 0.35, 0.0, 1.0)
    core_color = hsv_to_bgr(hue, max(0.0, sat * 0.75), core_val)
    cv2.circle(frame, center, radius, core_color, -1, cv2.LINE_AA)

    spec_alpha = int(120 * clamp(brightness, 0.0, 1.0))
    cv2.ellipse(
        frame,
        (center[0] - radius // 3, center[1] - radius // 3),
        (max(4, radius // 4), max(2, radius // 7)),
        -35,
        0,
        360,
        (spec_alpha, spec_alpha, spec_alpha),
        -1,
        cv2.LINE_AA,
    )

    dot_color = (0, 230, 0) if s["beat_flash"] > 0.05 else (60, 60, 60)
    cv2.circle(frame, (28, 28), 10, dot_color, -1, cv2.LINE_AA)

    x0, y0 = 20, 52
    bar_width, bar_height = 180, 12
    cv2.rectangle(frame, (x0, y0), (x0 + bar_width, y0 + bar_height), (50, 50, 50), 1)

    fill = int(bar_width * clamp(s["energy"], 0.0, 1.0))
    cv2.rectangle(frame, (x0 + 1, y0 + 1), (x0 + fill, y0 + bar_height - 1), (0, 180, 255), -1)
    put_text(frame, "ENERGY", (x0, y0 + bar_height + 18), 0.45, (200, 200, 200), 1)

    bpm_text = f"BPM {int(round(s['bpm']))}" if s["bpm"] > 1.0 else "BPM --"
    put_text(frame, bpm_text, (20, 105), 0.6, (235, 235, 235), 1)
    put_text(frame, f"Mode: {s['mode']}", (20, 130), 0.52, (200, 200, 200), 1)
    put_text(
        frame,
        f"Hand height: {s['height']:.2f}   Cycle: {s['cycle_time']:.2f}s",
        (20, 152),
        0.5,
        (200, 200, 200),
        1,
    )

    if s["raise_roof"]:
        put_text(frame, "RAISE THE ROOF -> STROBE", (20, 176), 0.55, (80, 220, 255), 1)

    if s["chill"] > 0.03:
        put_text(frame, f"CHILL-OUT {int(s['chill'] * 100)}%", (20, 198), 0.55, (120, 220, 160), 1)

    gauge_x = width - 46
    gauge_y = 60
    gauge_h = height - 160
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + 20, gauge_y + gauge_h), (60, 60, 60), 1)

    fill_h = int(gauge_h * clamp(s["height"], 0.0, 1.0))
    cv2.rectangle(
        frame,
        (gauge_x + 1, gauge_y + gauge_h - fill_h),
        (gauge_x + 19, gauge_y + gauge_h - 1),
        (0, 200, 220),
        -1,
    )
    put_text(frame, "HAND", (gauge_x - 14, gauge_y - 10), 0.42, (200, 200, 200), 1)

    put_text(frame, s["caption"][:95], (20, height - 20), 0.56, (240, 240, 240), 1)

    if s["event"]:
        put_text(frame, s["event"][:65], (width - 320, 30), 0.48, (160, 220, 255), 1)

    return frame


st.title("🪩 Gesture-Controlled Party Lamp")
st.caption(
    "Live audio-reactive lamp demo with beat detection, hand-height color control, swipe accents, "
    "raise-the-roof strobe, and chill-out wind-down. The demo starts automatically."
)

with st.sidebar:
    st.header("Controls")

    if "current_track" not in st.session_state:
        st.session_state.current_track = TRACK_OPTIONS[0]

    if "current_sens" not in st.session_state:
        st.session_state.current_sens = 1.35

    track_index = (
        TRACK_OPTIONS.index(st.session_state.current_track)
        if st.session_state.current_track in TRACK_OPTIONS
        else 0
    )

    selected_track = st.selectbox(
        "Music track",
        TRACK_OPTIONS,
        index=track_index,
        help="Steady beat for strong pulses, ambient track for graceful detector degradation.",
    )

    selected_sensitivity = st.slider(
        "Beat sensitivity threshold",
        min_value=1.05,
        max_value=2.50,
        value=float(st.session_state.current_sens),
        step=0.05,
        help="Lower values are more sensitive. Higher values reduce false triggers on quiet tracks.",
    )

    track_changed = selected_track != st.session_state.current_track
    sensitivity_changed = float(selected_sensitivity) != float(st.session_state.current_sens)

    if track_changed:
        st.session_state.current_track = selected_track
        st.session_state.last_beat_index = -1
        st.session_state.demo_start = time.time()

    if sensitivity_changed:
        st.session_state.current_sens = float(selected_sensitivity)
        st.session_state.last_beat_index = -1

    st.divider()

    auto_demo = st.toggle(
        "Auto 2-minute demo",
        value=bool(st.session_state.get("auto_demo", True)),
        help="Runs the scripted performance demo automatically.",
    )

    use_camera = st.toggle(
        "Live hand tracking (webcam)",
        value=bool(st.session_state.get("use_camera", False)),
        help="Uses MediaPipe Hands when available. Auto demo turns off when camera control is enabled.",
    )

    if use_camera and auto_demo:
        auto_demo = False

    if not use_camera:
        st.session_state.camera_error = None
        st.session_state.camera_raise = False

    st.session_state.auto_demo = auto_demo
    st.session_state.use_camera = use_camera

    st.divider()
    st.markdown("**Manual gesture simulation**")

    manual_disabled = auto_demo or use_camera

    manual_height = st.slider(
        "Hand height",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("manual_height", 0.15)),
        step=0.01,
        disabled=manual_disabled,
        help="Low hand = slow hue cycle. High hand = fast hue cycle.",
    )

    if not manual_disabled:
        st.session_state.manual_height = float(manual_height)

    if st.button("Swipe / color jump", disabled=bool(auto_demo)):
        st.session_state.pending_swipe = True
        st.session_state.last_event = "Manual swipe"

    manual_raise = st.toggle(
        "Raise the roof (manual hold)",
        value=bool(st.session_state.get("manual_raise", False)),
        disabled=manual_disabled,
    )
    st.session_state.manual_raise = bool(manual_raise)

    if st.button("Chill-out hold (1.5 s)", disabled=bool(auto_demo)):
        st.session_state.chill_hold_start = time.time()
        st.session_state.last_event = "Chill-out hold started"

    if st.button("Reset to party mode"):
        st.session_state.chill_target = 0.0
        st.session_state.chill_hold_start = None
        st.session_state.manual_raise = False
        st.session_state.auto_raise = False
        st.session_state.last_event = "Party reset"

    st.divider()

    if st.session_state.audio_backend == "fallback":
        st.caption("Browser audio fallback is active. Click play if sound does not start automatically.")
        if st.session_state.audio_error:
            st.caption(f"Pygame audio unavailable: {st.session_state.audio_error}")

        wav_bytes = get_track_assets(st.session_state.current_track)[3]
        st.audio(wav_bytes, format="audio/wav", autoplay=True)

if st.session_state.audio_backend is None or track_changed:
    start_audio(st.session_state.current_track)

beat_data = get_beat_data(st.session_state.current_track, st.session_state.current_sens)
st.session_state.beats, st.session_state.energies, st.session_state.energy_dt, st.session_state.sr = beat_data

if len(st.session_state.energies):
    st.session_state.energy_scale = float(np.max(st.session_state.energies))
else:
    st.session_state.energy_scale = 1.0

track_assets = get_track_assets(st.session_state.current_track)
st.session_state.duration = track_assets[4]

left_col, right_col = st.columns([2, 1])

video_placeholder = left_col.empty()
left_col.caption("Live demo feed — starts automatically and updates in real time.")

camera_placeholder = right_col.empty()
telemetry_placeholder = right_col.empty()


def live_view():
    now = time.time()
    dt = clamp(now - st.session_state.last_frame_time, 0.001, 0.25)
    st.session_state.last_frame_time = now

    beats = st.session_state.beats
    energies = st.session_state.energies
    energy_dt = st.session_state.energy_dt
    duration = st.session_state.duration

    pos = (now - st.session_state.play_start) % duration if st.session_state.play_start else 0.0

    auto_height = apply_auto_demo(now)
    camera_result = update_camera(now) if st.session_state.use_camera else None

    if camera_result is not None and camera_result.get("height") is not None:
        target_height = camera_result["height"]
        st.session_state.caption = (
            "Live hand tracking: height controls hue speed; swipe, raise, and chill gestures active."
        )
    elif auto_height is not None:
        target_height = auto_height
    else:
        target_height = st.session_state.manual_height
        if not st.session_state.caption or st.session_state.caption.startswith("Live hand tracking"):
            st.session_state.caption = "Manual mode: use sidebar controls or enable the auto demo."

    st.session_state.hand_height = lerp(
        st.session_state.hand_height,
        clamp(target_height, 0.0, 1.0),
        min(1.0, dt * 10.0),
    )
    hand_height = st.session_state.hand_height

    if st.session_state.pending_swipe:
        st.session_state.hue = random.random()
        st.session_state.flash = 1.0
        st.session_state.pending_swipe = False
        if not st.session_state.last_event:
            st.session_state.last_event = "Swipe color jump"

    if len(beats) > 0:
        idx = int(np.searchsorted(beats, pos, side="right")) - 1
        last_idx = st.session_state.last_beat_index

        if idx >= 0 and idx != last_idx:
            if idx < last_idx:
                last_idx = -1

            if pos - float(beats[idx]) <= 0.20:
                st.session_state.pulse = 1.0
                st.session_state.beat_flash = 1.0

            st.session_state.last_beat_index = idx
    else:
        st.session_state.last_beat_index = -1

    if st.session_state.chill_hold_start is not None:
        if now - st.session_state.chill_hold_start >= 1.5:
            st.session_state.chill_target = 1.0
            st.session_state.chill_hold_start = None
            st.session_state.last_event = "Chill-out activated"

    target_chill = clamp(st.session_state.chill_target, 0.0, 1.0)
    chill_rate = 0.9 if target_chill > st.session_state.chill else 1.4

    if st.session_state.chill < target_chill:
        st.session_state.chill = min(target_chill, st.session_state.chill + dt * chill_rate)
    else:
        st.session_state.chill = max(target_chill, st.session_state.chill - dt * chill_rate)

    party_cycle_time = lerp(10.0, 0.8, clamp(hand_height, 0.0, 1.0))
    cycle_time = lerp(party_cycle_time, 45.0, st.session_state.chill)
    st.session_state.hue = (st.session_state.hue + dt / max(0.1, cycle_time)) % 1.0

    st.session_state.pulse *= math.exp(-dt * 5.2)
    st.session_state.flash *= math.exp(-dt * 7.5)
    st.session_state.beat_flash *= math.exp(-dt * 6.0)

    raise_active = bool(
        st.session_state.manual_raise
        or st.session_state.camera_raise
        or st.session_state.auto_raise
    )

    strobe_enabled = raise_active and st.session_state.chill < 0.55
    beat_distance = nearest_beat_distance(beats, pos)
    strobe_on = strobe_enabled and (beat_distance < 0.10 or st.session_state.pulse > 0.82)

    if len(energies) > 0:
        energy_index = int(pos / max(1e-6, energy_dt)) % len(energies)
        energy = float(energies[energy_index])
        energy_norm = clamp(energy / max(1e-6, st.session_state.energy_scale), 0.0, 1.0)
    else:
        energy_norm = 0.0

    bpm = bpm_estimate(beats, pos)

    if st.session_state.chill > 0.55:
        mode = "CHILL"
    elif strobe_enabled:
        mode = "STROBE"
    else:
        mode = "PARTY"

    if st.session_state.auto_demo:
        mode = "AUTO " + mode
    elif st.session_state.use_camera:
        mode = "CAMERA " + mode

    if not st.session_state.auto_demo and not st.session_state.use_camera:
        if strobe_enabled:
            st.session_state.caption = "Raise-the-roof held: beat-synced strobe."
        elif st.session_state.chill > 0.5:
            st.session_state.caption = "Chill-out: slow single-color glow."
        else:
            st.session_state.caption = "Manual performance mode."

    sat = lerp(0.95, 0.35, st.session_state.chill)
    val = lerp(0.95, 0.55, st.session_state.chill)

    render_state = {
        "hue": st.session_state.hue,
        "sat": sat,
        "val": val,
        "pulse": st.session_state.pulse,
        "flash": st.session_state.flash,
        "strobe_active": strobe_enabled,
        "strobe_on": strobe_on,
        "chill": st.session_state.chill,
        "beat_flash": st.session_state.beat_flash,
        "energy": energy_norm,
        "bpm": bpm,
        "height": hand_height,
        "cycle_time": cycle_time,
        "caption": st.session_state.caption,
        "mode": mode,
        "raise_roof": raise_active,
        "event": st.session_state.last_event,
    }

    frame = render_frame(render_state)
    video_placeholder.image(frame, channels="BGR", use_container_width=True)

    if st.session_state.use_camera:
        if st.session_state.camera_frame is not None:
            camera_placeholder.image(
                st.session_state.camera_frame,
                channels="BGR",
                use_container_width=True,
            )
        elif st.session_state.camera_error:
            camera_placeholder.warning(st.session_state.camera_error)
        else:
            camera_placeholder.info("Starting camera...")
    else:
        camera_placeholder.empty()

    beat_indicator = "🔴" if st.session_state.beat_flash > 0.1 else "⚪"
    energy_bars = int(energy_norm * 10)
    energy_display = "▓" * energy_bars + "░" * (10 - energy_bars)

    telemetry_placeholder.markdown(
        f"""
        | Metric | Value |
        |---|---|
        | BPM | {int(round(bpm)) if bpm > 1.0 else "--"} |
        | Beat flash | {beat_indicator} |
        | Energy | {energy_display} |
        | Mode | {mode} |
        | Hand height | {hand_height:.2f} |
        | Cycle time | {cycle_time:.2f} s |
        | Last event | {st.session_state.last_event or "—"} |
        """
    )


if hasattr(st, "fragment"):
    live_view = st.fragment(run_every=FRAME_DT)(live_view)

live_view()

if not hasattr(st, "fragment"):
    time.sleep(FRAME_DT)
    st.rerun()