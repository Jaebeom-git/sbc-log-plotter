# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""Shared data model and signal constants for log plotting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


JOINT_COUNT = 27
ANKLE_AXIS_COUNT = 4
ANKLE_AXIS_NAMES = [
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_pitch",
    "right_ankle_roll",
]
SBC_JOINT_FIELDS = [
    "ref_pos",
    "ref_vel",
    "ref_kp",
    "ref_kd",
    "ref_torq",
    "out_torq",
    "act_pos",
    "act_vel",
    "act_torq",
    "status_word",
    "torq_cmd_pdo",
]
SBC_PLOT_FIELDS = [field for field in SBC_JOINT_FIELDS if field != "status_word"]
SBC_ANKLE_FIELDS = ["ankle_ref_pos", "ankle_ref_vel", "ankle_act_pos", "ankle_act_vel"]
SBC_EXPECTED_COLUMNS = (
    JOINT_COUNT * len(SBC_JOINT_FIELDS)
    + ANKLE_AXIS_COUNT * len(SBC_ANKLE_FIELDS)
)
FIELD_COLORS = {
    "ref_pos": (255, 140, 50),
    "ref_vel": (255, 190, 70),
    "ref_kp": (210, 160, 255),
    "ref_kd": (180, 120, 255),
    "ref_torq": (255, 95, 95),
    "out_torq": (80, 140, 255),
    "act_pos": (50, 220, 110),
    "act_vel": (220, 220, 90),
    "act_torq": (70, 210, 220),
    "torq_cmd_pdo": (240, 240, 240),
}
FIELD_AXIS_LABELS = {
    "ref_pos": "ref_pos [rad]",
    "ref_vel": "ref_vel [rad/s]",
    "ref_kp": "ref_kp",
    "ref_kd": "ref_kd",
    "ref_torq": "ref_torq",
    "out_torq": "out_torq",
    "act_pos": "act_pos [rad]",
    "act_vel": "act_vel [rad/s]",
    "act_torq": "act_torq",
    "torq_cmd_pdo": "torq_cmd_pdo",
}


@dataclass(frozen=True)
class MotorLayout:
    names: list[str]
    groups: dict[str, list[int]]


@dataclass(frozen=True)
class LogData:
    time: np.ndarray
    joint_names: list[str]
    fields: dict[str, np.ndarray]
    ankle_fields: dict[str, np.ndarray]
    sample_rate_hz: float
    column_headers: list[str]
    source: str = "sbc"
    limits: dict[str, np.ndarray] = field(default_factory=dict)


# Compatibility name used by existing plotting/tests.
SbcLogData = LogData
