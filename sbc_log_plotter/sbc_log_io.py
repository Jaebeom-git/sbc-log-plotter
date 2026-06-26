# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""SBC text log IO.

This module is intentionally limited to SBC flat text logs and YAML-based
motor layout discovery so it can be shared without simulation package deps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .model import (
    ANKLE_AXIS_COUNT,
    ANKLE_AXIS_NAMES,
    FIELD_AXIS_LABELS,
    FIELD_COLORS,
    JOINT_COUNT,
    MotorLayout,
    SBC_ANKLE_FIELDS,
    SBC_EXPECTED_COLUMNS,
    SBC_JOINT_FIELDS,
    SBC_PLOT_FIELDS,
    SbcLogData,
)


def joint_names(count: int = JOINT_COUNT) -> list[str]:
    return [f"J{i:02d}" for i in range(1, count + 1)]


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data if isinstance(data, dict) else {}


def _insert_after(values: list[str], existing: str, inserted: str) -> None:
    if inserted in values:
        return
    try:
        values.insert(values.index(existing) + 1, inserted)
    except ValueError:
        values.append(inserted)


def _load_upper_motor_names(repo_root: Path) -> list[str]:
    arm_config = repo_root / "robots" / "kimm_P2" / "arm_config.yaml"
    names: list[str] = []
    if arm_config.exists():
        raw = _read_yaml(arm_config)
        joints = raw.get("joints", [])
        if isinstance(joints, list):
            for joint in joints:
                if isinstance(joint, dict) and joint.get("actuator_name"):
                    names.append(str(joint["actuator_name"]))

    urdf = repo_root / "robots" / "P_2_Robot" / "P_2_Robot_meshOpt.urdf"
    if urdf.exists():
        text = urdf.read_text(encoding="utf-8", errors="ignore")
        if "left_wrist_roll_joint" in text:
            _insert_after(names, "left_wrist_pitch", "left_wrist_roll")
        if "right_wrist_roll_joint" in text:
            _insert_after(names, "right_wrist_pitch", "right_wrist_roll")

    if len(names) < 14:
        fallback_config = repo_root / "robots" / "kimm_P1" / "arm_config.yaml"
        if fallback_config.exists():
            raw = _read_yaml(fallback_config)
            fallback = [
                str(joint["actuator_name"])
                for joint in raw.get("joints", [])
                if isinstance(joint, dict) and joint.get("actuator_name")
            ]
            if len(fallback) >= 14:
                names = fallback

    names = names[:14]
    while len(names) < 14:
        names.append(f"upper_unknown_{len(names) + 1:02d}")
    return names


def _policy_joint_to_motor_name(policy_name: str) -> str:
    name = str(policy_name).strip()
    canonical = name.upper()
    if canonical == "WAIST":
        return "waist_yaw"

    for prefix, side in (("L_", "left_"), ("R_", "right_")):
        if not canonical.startswith(prefix):
            continue
        suffix = canonical[len(prefix):].lower()
        if suffix.startswith("ank_"):
            suffix = f"ankle_{suffix[len('ank_'):]}"
        return f"{side}{suffix}"

    return name.lower()


def _lower_names_from_policy_yaml(policy_config: Path) -> list[str] | None:
    if not policy_config.is_file():
        return None
    raw = _read_yaml(policy_config)
    robot_order = raw.get("joint_order", {}).get("robot", [])
    if isinstance(robot_order, list) and robot_order:
        return [_policy_joint_to_motor_name(str(name)) for name in robot_order]
    return None


def _sbc_log_config_from_policy_yaml(policy_config: Path | None) -> dict:
    if policy_config is None or not policy_config.is_file():
        return {}
    raw = _read_yaml(policy_config)
    config = raw.get("sbc_log", {})
    return config if isinstance(config, dict) else {}


def _int_config_value(config: dict, key: str) -> int | None:
    if key not in config:
        return None
    value = config[key]
    if isinstance(value, bool):
        raise ValueError(f"sbc_log.{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sbc_log.{key} must be an integer") from exc


def _load_lower_motor_names(repo_root: Path, policy_yaml: Path | None = None) -> list[str]:
    # Prefer an explicitly provided policy YAML (e.g. selected in the GUI),
    # then fall back to the conventional repo-relative config path.
    for candidate in (policy_yaml, repo_root / "config" / "policy_p2_new_scaled.yaml"):
        if candidate is None:
            continue
        names = _lower_names_from_policy_yaml(Path(candidate))
        if names is not None:
            return names
    return [
        "waist_yaw",
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
    ]


def _layout_from_sbc_log_config(
    *,
    total_motor_count: int,
    lower_start_index: int,
    lower_names: list[str],
) -> list[str]:
    if total_motor_count <= 0:
        raise ValueError("sbc_log.total_motor_count must be positive")
    if lower_start_index < 0:
        raise ValueError("sbc_log.lower_start_index must be non-negative")
    if lower_start_index + len(lower_names) > total_motor_count:
        raise ValueError(
            "sbc_log.lower_start_index plus joint_order.robot length exceeds "
            "sbc_log.total_motor_count"
        )

    names = joint_names(total_motor_count)
    for offset, motor_name in enumerate(lower_names):
        names[lower_start_index + offset] = motor_name
    return names


def _default_plot_groups(names: list[str]) -> dict[str, list[int]]:
    lower_body_layout = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
    ]
    lower_indices = [names.index(name) for name in lower_body_layout if name in names]
    return {
        "Lower Body": lower_indices,
        "Upper Body": [idx for idx in range(0, min(14, len(names))) if idx not in lower_indices],
        "All": list(range(0, len(names))),
    }


def _plot_groups_from_sbc_log_config(config: dict, names: list[str]) -> dict[str, list[int]] | None:
    raw_groups = config.get("plot_groups")
    if raw_groups is None:
        return None
    if not isinstance(raw_groups, dict):
        raise ValueError("sbc_log.plot_groups must be a mapping")

    by_name = {name: idx for idx, name in enumerate(names)}
    groups: dict[str, list[int]] = {}
    for group_name, raw_joint_names in raw_groups.items():
        if not isinstance(raw_joint_names, list):
            raise ValueError(f"sbc_log.plot_groups.{group_name} must be a list")
        indices: list[int] = []
        for raw_joint_name in raw_joint_names:
            raw_name = str(raw_joint_name)
            joint_name = raw_name if raw_name in by_name else _policy_joint_to_motor_name(raw_name)
            if joint_name not in by_name:
                raise ValueError(f"sbc_log.plot_groups.{group_name} references unknown joint: {raw_joint_name}")
            indices.append(by_name[joint_name])
        groups[str(group_name)] = indices

    groups.setdefault("All", list(range(0, len(names))))
    if "Upper Body" not in groups:
        grouped = {idx for group_name, indices in groups.items() if group_name != "All" for idx in indices}
        groups["Upper Body"] = [idx for idx in range(0, len(names)) if idx not in grouped]
    return groups


def default_motor_layout(
    repo_root: str | Path | None = None,
    policy_yaml: str | Path | None = None,
) -> MotorLayout:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    policy_yaml_path = Path(policy_yaml) if policy_yaml is not None else None
    lower_names = _load_lower_motor_names(root, policy_yaml_path)

    sbc_log_config = _sbc_log_config_from_policy_yaml(policy_yaml_path)
    total_motor_count = _int_config_value(sbc_log_config, "total_motor_count")
    lower_start_index = _int_config_value(sbc_log_config, "lower_start_index")
    has_sbc_log_layout = total_motor_count is not None or lower_start_index is not None
    if has_sbc_log_layout:
        if total_motor_count is None:
            total_motor_count = JOINT_COUNT
        if lower_start_index is None:
            lower_start_index = 14
        names = _layout_from_sbc_log_config(
            total_motor_count=total_motor_count,
            lower_start_index=lower_start_index,
            lower_names=lower_names,
        )
    else:
        upper_names = _load_upper_motor_names(root)
        names = [*upper_names, *lower_names]

    if not has_sbc_log_layout and len(names) != JOINT_COUNT:
        names = [*names[:JOINT_COUNT], *joint_names(JOINT_COUNT)[len(names):]]
    groups = _plot_groups_from_sbc_log_config(sbc_log_config, names) or _default_plot_groups(names)
    return MotorLayout(
        names=names,
        groups=groups,
    )


def plot_indices_for_group(layout: MotorLayout, group_name: str) -> list[int]:
    try:
        return list(layout.groups[group_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported motor group: {group_name}") from exc


def sbc_column_headers(motor_layout: MotorLayout | None = None) -> list[str]:
    layout = motor_layout or default_motor_layout()

    headers: list[str] = []
    for joint_name in layout.names:
        headers.extend(f"{joint_name}.{field_name}" for field_name in SBC_JOINT_FIELDS)
    for ankle_name in ANKLE_AXIS_NAMES:
        headers.extend(f"{ankle_name}.{field_name}" for field_name in SBC_ANKLE_FIELDS)
    return headers


def sbc_cache_path(path: str | Path) -> Path:
    log_path = Path(path)
    return log_path.with_name(f".{log_path.name}.sbc-cache.npz")


def _robot_joint_name_for_limit(motor_name: str) -> str:
    return motor_name if motor_name.endswith("_joint") else f"{motor_name}_joint"


def optional_robot_limit_fields(robot: str | None, motor_layout: MotorLayout) -> dict[str, np.ndarray]:
    return {}


def _parse_flat_text(log_path: Path) -> np.ndarray:
    try:
        text = log_path.read_bytes()
        # np.fromstring handles generic whitespace faster than np.fromfile(sep)
        # on many large text logs. Replacing tabs with spaces keeps parsing
        # predictable for logs that mix tabs and newlines.
        flat = np.fromstring(text.replace(b"\t", b" "), dtype=float, sep=" ")
        if flat.size > 0:
            return flat
    except Exception:
        pass
    return np.fromfile(log_path, dtype=float, sep="\t")


def _load_flat_values(log_path: Path) -> np.ndarray:
    return _parse_flat_text(log_path)


def sbc_expected_columns(motor_count: int = JOINT_COUNT) -> int:
    return motor_count * len(SBC_JOINT_FIELDS) + ANKLE_AXIS_COUNT * len(SBC_ANKLE_FIELDS)


def load_sbc_log(
    path: str | Path,
    sample_rate_hz: float = 1000.0,
    time_offset: float = 0.0,
    motor_layout: MotorLayout | None = None,
    limit_robot: str | None = None,
) -> SbcLogData:
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")

    log_path = Path(path)
    if not log_path.is_file():
        raise FileNotFoundError(f"SBC log path not found: {log_path}")

    layout = motor_layout or default_motor_layout()
    motor_count = len(layout.names)
    expected_columns = sbc_expected_columns(motor_count)

    flat = _load_flat_values(log_path)
    if flat.size == 0:
        raise ValueError(f"No numeric samples found in {log_path}")
    if flat.size % expected_columns != 0:
        raise ValueError(
            f"Expected {expected_columns} tab-separated values per row, "
            f"but found {flat.size} total values"
        )

    raw = flat.reshape((-1, expected_columns))
    joint_raw = raw[:, : motor_count * len(SBC_JOINT_FIELDS)].reshape(
        raw.shape[0], motor_count, len(SBC_JOINT_FIELDS)
    )
    ankle_raw = raw[:, motor_count * len(SBC_JOINT_FIELDS):].reshape(
        raw.shape[0], ANKLE_AXIS_COUNT, len(SBC_ANKLE_FIELDS)
    )

    fields = {
        field_name: joint_raw[:, :, field_idx]
        for field_idx, field_name in enumerate(SBC_JOINT_FIELDS)
    }
    ankle_fields = {
        field_name: ankle_raw[:, :, field_idx]
        for field_idx, field_name in enumerate(SBC_ANKLE_FIELDS)
    }
    time = time_offset + np.arange(raw.shape[0], dtype=float) / sample_rate_hz

    return SbcLogData(
        time=time,
        joint_names=layout.names,
        fields=fields,
        ankle_fields=ankle_fields,
        sample_rate_hz=sample_rate_hz,
        column_headers=sbc_column_headers(layout),
        source="sbc",
        limits=optional_robot_limit_fields(limit_robot, layout),
    )
