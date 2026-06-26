# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""GUI-free signal catalog and display options for log plotting."""
from __future__ import annotations

from . import model

DERIVED_MOTOR_FIELDS = [
    "motor_ref_pos",
    "motor_act_pos",
    "motor_act_vel",
    "motor_ref_torq",
    "motor_act_torq",
    "default_q",
    "scaled_action",
]

SIGNAL_LABEL_BY_FIELD: dict[str, str] = {
    "None": "None",
    "ref_pos": "Reference position [rad]",
    "ref_vel": "Reference velocity [rad/s]",
    "ref_kp": "Reference Kp",
    "ref_kd": "Reference Kd",
    "ref_torq": "Reference torque [Nm]",
    "out_torq": "Output torque [Nm]",
    "act_pos": "Actual position [rad]",
    "act_vel": "Actual velocity [rad/s]",
    "act_torq": "Actual torque [Nm]",
    "torq_cmd_pdo": "Torque command (PDO) [Nm]",
    "motor_ref_pos": "Motor reference position [rad]",
    "motor_act_pos": "Motor actual position [rad]",
    "motor_act_vel": "Motor actual velocity [rad/s]",
    "motor_ref_torq": "Motor reference torque [Nm]",
    "motor_act_torq": "Motor actual torque [Nm]",
    "default_q": "Default position [rad]",
    "scaled_action": "Scaled action [rad]",
}
SIGNAL_FIELD_BY_LABEL: dict[str, str] = {
    label: field for field, label in SIGNAL_LABEL_BY_FIELD.items()
}
SIGNAL_COLORS: dict[str, tuple[int, int, int]] = {
    **model.FIELD_COLORS,
    "motor_ref_pos": (255, 180, 95),
    "motor_act_pos": (95, 255, 155),
    "motor_act_vel": (200, 200, 130),
    "motor_ref_torq": (255, 115, 115),
    "motor_act_torq": (90, 210, 220),
    "default_q": (160, 160, 160),
    "scaled_action": (106, 61, 154),
}
ALL_SIGNAL_FIELDS = ["None", *model.SBC_PLOT_FIELDS, *DERIVED_MOTOR_FIELDS]
SLOT_OPTIONS = [SIGNAL_LABEL_BY_FIELD[f] for f in ALL_SIGNAL_FIELDS]

SIGNAL_UNIT_BY_FIELD: dict[str, str] = {
    "ref_pos": "rad", "act_pos": "rad",
    "motor_ref_pos": "rad", "motor_act_pos": "rad",
    "ref_vel": "rad/s", "act_vel": "rad/s", "motor_act_vel": "rad/s",
    "ref_torq": "Nm", "out_torq": "Nm", "act_torq": "Nm",
    "motor_ref_torq": "Nm", "motor_act_torq": "Nm",
    "default_q": "rad", "scaled_action": "rad",
    "ref_kp": "", "ref_kd": "",
    "torq_cmd_pdo": "pdo",
}

FILTER_CHOICES: list[tuple[str, str]] = [
    ("None", "none"),
    ("Moving Average", "ma"),
    ("Low-pass filter", "lpf"),
    ("High-pass filter", "hpf"),
]
FILTER_LABEL_TO_KIND = {label: kind for label, kind in FILTER_CHOICES}
FILTER_KIND_TO_LABEL = {kind: label for label, kind in FILTER_CHOICES}

PHASE_CHOICES: list[tuple[str, str]] = [
    ("Zero-phase (filtfilt)", "zero"),
    ("Causal (sosfilt)", "causal"),
]
PHASE_LABEL_TO_VALUE = {label: value for label, value in PHASE_CHOICES}
PHASE_VALUE_TO_LABEL = {value: label for label, value in PHASE_CHOICES}

FFT_WINDOW_OPTIONS = ["hann", "hamming", "rect"]
FFT_SCALE_OPTIONS = ["linear", "db"]
FFT_DETREND_OPTIONS = ["none", "mean"]
LINE_STYLE_CHOICES: list[tuple[str, str]] = [
    ("Solid (-)", "-"),
    ("Dashed (--)", "--"),
    ("Dash-dot (-.)", "-."),
    ("Dotted (:)", ":"),
]
LINE_STYLE_BY_LABEL = {label: style for label, style in LINE_STYLE_CHOICES}
LINE_STYLE_LABEL_BY_STYLE = {style: label for label, style in LINE_STYLE_CHOICES}
COLOR_CHOICES: list[tuple[str, tuple[int, int, int] | None]] = [
    ("Default", None),
    ("Blue", (0, 87, 184)),
    ("Red", (212, 17, 89)),
    ("Green", (26, 152, 80)),
    ("Orange", (255, 127, 0)),
    ("Purple", (106, 61, 154)),
    ("Cyan", (0, 166, 214)),
    ("Gray", (77, 77, 77)),
    ("White", (240, 240, 240)),
]
COLOR_BY_LABEL = {label: color for label, color in COLOR_CHOICES}
SAMPLE_COLOR_CHOICES: list[tuple[str, tuple[int, int, int] | None]] = [
    ("Complement", None),
    *[(label, color) for label, color in COLOR_CHOICES if color is not None],
]
SAMPLE_COLOR_BY_LABEL = {label: color for label, color in SAMPLE_COLOR_CHOICES}
PNG_SIZE_CHOICES: list[tuple[str, tuple[int, int] | None]] = [
    ("4:3 QXGA (2048×1536)", (2048, 1536)),
    ("4:3 UXGA (1600×1200)", (1600, 1200)),
    ("16:9 FHD (1920×1080)", (1920, 1080)),
    ("16:9 QHD (2560×1440)", (2560, 1440)),
    ("1:1 Square (1600×1600)", (1600, 1600)),
    ("Custom", None),
]
PNG_SIZE_BY_LABEL = {label: size for label, size in PNG_SIZE_CHOICES}


def signal_unit(signal: str) -> str:
    return "" if signal == "None" else SIGNAL_UNIT_BY_FIELD.get(signal, "")
