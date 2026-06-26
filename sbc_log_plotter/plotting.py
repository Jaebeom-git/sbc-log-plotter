# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""GUI-free plotting data helpers for log_plot.

No Qt or matplotlib imports live here. These helpers are shared by the GUI,
batch PNG export, and save-mode YAML path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import model
from . import signal_pipeline as sp
from .catalog import (
    ALL_SIGNAL_FIELDS,
    SIGNAL_COLORS,
    SIGNAL_LABEL_BY_FIELD,
    signal_unit,
)
from .state import SlotState, ViewMode

_ANKLE_BLOCK_COLS = {
    "left_ankle_pitch": 0,
    "left_ankle_roll": 1,
    "right_ankle_pitch": 2,
    "right_ankle_roll": 3,
}
_ANKLE_FIELD_MAP = {
    "ref_pos": "ankle_ref_pos",
    "ref_vel": "ankle_ref_vel",
    "act_pos": "ankle_act_pos",
    "act_vel": "ankle_act_vel",
}

_LOWER_BODY_PLOT_ORDER = [
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


@dataclass(frozen=True)
class LimitLine:
    axis: Literal["x", "y"]
    value: float
    label: str
    color: tuple[int, int, int]
    style: str = "--"


@dataclass(frozen=True)
class PlotCell:
    joint_name: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


def build_signal_fields(data: model.SbcLogData) -> dict[str, np.ndarray]:
    fields = {
        f: np.asarray(data.fields[f], dtype=float).copy()
        for f in model.SBC_PLOT_FIELDS
        if f in data.fields
    }
    motor_fallbacks = {
        "motor_ref_pos": "ref_pos",
        "motor_act_pos": "act_pos",
        "motor_act_vel": "act_vel",
        "motor_ref_torq": "ref_torq",
        "motor_act_torq": "act_torq",
    }
    for motor_field, fallback_field in motor_fallbacks.items():
        if motor_field in data.fields:
            source = data.fields[motor_field]
        elif data.source == "sbc":
            source = data.fields.get(fallback_field)
        else:
            source = None
        if source is not None:
            fields[motor_field] = np.asarray(source, dtype=float).copy()
    for optional_field in ("default_q", "scaled_action"):
        if optional_field in data.fields:
            fields[optional_field] = np.asarray(data.fields[optional_field], dtype=float).copy()

    for joint_name, col in _ANKLE_BLOCK_COLS.items():
        if joint_name not in data.joint_names:
            continue
        joint_idx = data.joint_names.index(joint_name)
        for base_field, ankle_field in _ANKLE_FIELD_MAP.items():
            if base_field not in fields:
                continue
            ankle_values = data.ankle_fields.get(ankle_field)
            if ankle_values is None or ankle_values.shape[1] <= col:
                continue
            fields[base_field][:, joint_idx] = ankle_values[:, col]
    return fields


def field_has_available_samples(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=float)
    return arr.size > 0 and np.isfinite(arr).any()


def slot_options_for_signal_fields(signal_fields: dict[str, np.ndarray]) -> list[str]:
    options = [SIGNAL_LABEL_BY_FIELD["None"]]
    for field_name in ALL_SIGNAL_FIELDS:
        if field_name == "None" or field_name not in signal_fields:
            continue
        if field_has_available_samples(signal_fields[field_name]):
            options.append(SIGNAL_LABEL_BY_FIELD[field_name])
    return options


def enabled_slot_indices(view_mode: ViewMode) -> list[int]:
    return [0, 1] if view_mode == "scatter" else [0, 1, 2]


def slot_color_rgb(slot_state: SlotState) -> tuple[int, int, int]:
    if slot_state.color_override is not None:
        return slot_state.color_override
    return SIGNAL_COLORS.get(slot_state.signal, (255, 255, 255))


def rgb_to_float(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return color[0] / 255.0, color[1] / 255.0, color[2] / 255.0


def slot_color_float(slot_state: SlotState) -> tuple[float, float, float]:
    return rgb_to_float(slot_color_rgb(slot_state))


def joint_title(joint_name: str) -> str:
    return joint_name.replace("_joint", "").replace("_", " ").title()


def lower_body_names_from_joint_names(joint_names: list[str]) -> list[str]:
    by_base = {name.removesuffix("_joint"): name for name in joint_names}
    ordered = [by_base[name] for name in _LOWER_BODY_PLOT_ORDER if name in by_base]
    if ordered:
        return ordered
    return [name for name in joint_names if any(part in name for part in ("hip", "knee", "ankle"))]


_UPPER_BODY_PLOT_ORDER = [
    "left_shoulder_yaw",
    "left_shoulder_roll",
    "left_shoulder_pitch",
    "right_shoulder_yaw",
    "right_shoulder_roll",
    "right_shoulder_pitch",
    "left_elbow",
    "right_elbow",
    "left_wrist_yaw",
    "left_wrist_pitch",
    "left_wrist_roll",
    "right_wrist_yaw",
    "right_wrist_pitch",
    "right_wrist_roll",
]

_WAIST_PLOT_ORDER = ["waist_yaw", "waist_roll", "waist_pitch"]


def _ordered_names_by_base(joint_names: list[str], order: list[str]) -> list[str]:
    by_base = {name.removesuffix("_joint"): name for name in joint_names}
    return [by_base[name] for name in order if name in by_base]


def upper_body_names_from_joint_names(joint_names: list[str]) -> list[str]:
    ordered = _ordered_names_by_base(joint_names, _UPPER_BODY_PLOT_ORDER)
    if ordered:
        return ordered
    return [name for name in joint_names if any(part in name for part in ("shoulder", "elbow", "wrist"))]


def waist_names_from_joint_names(joint_names: list[str]) -> list[str]:
    ordered = _ordered_names_by_base(joint_names, _WAIST_PLOT_ORDER)
    if ordered:
        return ordered
    return [name for name in joint_names if "waist" in name]


def plot_groups_from_joint_names(joint_names: list[str]) -> dict[str, list[str]]:
    lower = lower_body_names_from_joint_names(joint_names)
    waist = waist_names_from_joint_names(joint_names)
    upper = upper_body_names_from_joint_names(joint_names)
    grouped = set(lower) | set(waist) | set(upper)
    rest = [name for name in joint_names if name not in grouped]
    return {
        "Lower Body": lower,
        "Upper Body": upper,
        "Waist": waist,
        "All": [*lower, *waist, *upper, *rest],
    }


def lower_body_names_from_layout(layout: model.MotorLayout) -> list[str]:
    lower_indices = layout.groups.get("Lower Body", [])
    return [layout.names[i] for i in lower_indices]


def plot_groups_from_layout(layout: model.MotorLayout) -> dict[str, list[str]]:
    groups = {
        group_name: [layout.names[i] for i in indices if 0 <= i < len(layout.names)]
        for group_name, indices in layout.groups.items()
    }
    inferred = plot_groups_from_joint_names(layout.names)
    for group_name in ("Lower Body", "Upper Body", "Waist"):
        groups.setdefault(group_name, inferred[group_name])
    groups["All"] = [name for name in [*groups["Lower Body"], *groups["Waist"], *groups["Upper Body"]] if name in layout.names]
    seen = set(groups["All"])
    groups["All"].extend(name for name in layout.names if name not in seen)
    return groups


def _flat_plot_cells(joint_names: list[str], *, start_row: int = 0, tile_cols: int = 3) -> list[PlotCell]:
    return [
        PlotCell(joint_name=name, row=start_row + idx // tile_cols, col=idx % tile_cols)
        for idx, name in enumerate(joint_names)
    ]


def _semantic_plot_cells(joint_names: list[str], slots: list[tuple[str, int, int]]) -> list[PlotCell]:
    by_base = {name.removesuffix("_joint"): name for name in joint_names}
    cells = [
        PlotCell(joint_name=by_base[base], row=row, col=col)
        for base, row, col in slots
        if base in by_base
    ]
    return cells


_LOWER_BODY_CELLS = [
    ("left_hip_yaw", 0, 0),
    ("left_hip_roll", 0, 1),
    ("left_hip_pitch", 0, 2),
    ("right_hip_yaw", 1, 0),
    ("right_hip_roll", 1, 1),
    ("right_hip_pitch", 1, 2),
    ("left_knee", 2, 0),
    ("left_ankle_pitch", 2, 1),
    ("left_ankle_roll", 2, 2),
    ("right_knee", 3, 0),
    ("right_ankle_pitch", 3, 1),
    ("right_ankle_roll", 3, 2),
]

_UPPER_BODY_CELLS = [
    ("left_shoulder_yaw", 0, 0),
    ("left_shoulder_roll", 0, 1),
    ("left_shoulder_pitch", 0, 2),
    ("right_shoulder_yaw", 1, 0),
    ("right_shoulder_roll", 1, 1),
    ("right_shoulder_pitch", 1, 2),
    ("left_elbow", 2, 0),
    ("right_elbow", 3, 0),
    ("left_wrist_yaw", 4, 0),
    ("left_wrist_pitch", 4, 1),
    ("left_wrist_roll", 4, 2),
    ("right_wrist_yaw", 5, 0),
    ("right_wrist_pitch", 5, 1),
    ("right_wrist_roll", 5, 2),
]

_WAIST_CELLS = [
    ("waist_yaw", 0, 0),
    ("waist_roll", 0, 1),
    ("waist_pitch", 0, 2),
]


def plot_grid_shape_for_cells(cells: list[PlotCell]) -> tuple[int, int]:
    if not cells:
        return 1, 1
    rows = max(cell.row + cell.rowspan for cell in cells)
    cols = max(cell.col + cell.colspan for cell in cells)
    return max(1, rows), max(1, cols)


def plot_cells_for_group(group_name: str, joint_names: list[str]) -> list[PlotCell]:
    if group_name == "Lower Body":
        cells = _semantic_plot_cells(joint_names, _LOWER_BODY_CELLS)
        return cells if len(cells) == len(joint_names) else _flat_plot_cells(joint_names)
    if group_name == "Upper Body":
        cells = _semantic_plot_cells(joint_names, _UPPER_BODY_CELLS)
        return cells or _flat_plot_cells(joint_names)
    if group_name == "Waist":
        cells = _semantic_plot_cells(joint_names, _WAIST_CELLS)
        return cells or _flat_plot_cells(joint_names)
    if group_name == "All":
        groups = plot_groups_from_joint_names(joint_names)
        cells: list[PlotCell] = []
        row_offset = 0
        for section_name in ("Lower Body", "Waist", "Upper Body"):
            section_cells = plot_cells_for_group(section_name, groups[section_name])
            if not section_cells:
                continue
            cells.extend(
                PlotCell(
                    joint_name=cell.joint_name,
                    row=cell.row + row_offset,
                    col=cell.col,
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                )
                for cell in section_cells
            )
            row_offset += plot_grid_shape_for_cells(section_cells)[0]
        plotted = {cell.joint_name for cell in cells}
        rest = [name for name in joint_names if name not in plotted]
        cells.extend(_flat_plot_cells(rest, start_row=row_offset))
        return cells
    return _flat_plot_cells(joint_names)


def plot_grid_shape(item_count: int, tile_cols: int = 3) -> tuple[int, int]:
    if item_count <= 1:
        return 1, 1
    cols = max(1, min(int(tile_cols), item_count))
    rows = int(np.ceil(item_count / cols))
    return rows, cols


def apply_time_range(times: np.ndarray, values: np.ndarray, time_range: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    if time_range is None:
        return x, y
    start, end = time_range
    if end < start:
        start, end = end, start
    mask = (x >= float(start)) & (x <= float(end))
    return x[mask], y[mask]


def sample_indices_for_hz(times: np.ndarray, sample_hz: float | None) -> np.ndarray | None:
    if sample_hz is None or sample_hz <= 0.0 or times.size < 2:
        return None
    dt = np.diff(np.asarray(times, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return None
    fs = 1.0 / float(np.median(dt))
    stride = max(1, int(round(fs / float(sample_hz))))
    return np.arange(0, times.size, stride, dtype=int)


def sample_xy_for_hz(times: np.ndarray, values: np.ndarray, sample_hz: float | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    indices = sample_indices_for_hz(x, sample_hz)
    if indices is None:
        return np.array([]), np.array([])
    return x[indices], y[indices]


def zero_center_range(values: np.ndarray, margin: float = 0.05) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    radius = float(np.max(np.abs(finite)))
    if radius <= 0.0:
        radius = 1.0
    else:
        radius *= 1.0 + float(margin)
    return -radius, radius


def _limit_value(limits: dict[str, np.ndarray], key: str, joint_idx: int) -> float | None:
    values = limits.get(key)
    if values is None or joint_idx >= len(values):
        return None
    value = float(np.asarray(values, dtype=float)[joint_idx])
    return value if np.isfinite(value) else None


def limit_lines_for_signal(signal: str, joint_idx: int, limits: dict[str, np.ndarray], *, axis: Literal["x", "y"] = "y") -> list[LimitLine]:
    lines: list[LimitLine] = []
    grey = (107, 114, 128)
    if signal in {"ref_torq", "out_torq", "act_torq", "motor_ref_torq", "motor_act_torq"}:
        value = _limit_value(limits, "torque", joint_idx)
        if value is not None:
            lines.extend([LimitLine(axis, abs(value), "Torque limit", grey), LimitLine(axis, -abs(value), "_nolegend_", grey)])
    elif signal in {"ref_vel", "act_vel", "motor_act_vel"}:
        value = _limit_value(limits, "velocity", joint_idx)
        if value is not None:
            lines.extend([LimitLine(axis, abs(value), "Velocity limit", grey), LimitLine(axis, -abs(value), "_nolegend_", grey)])
    elif signal in {"ref_pos", "act_pos", "motor_ref_pos", "motor_act_pos"}:
        pos_min = _limit_value(limits, "position_min", joint_idx)
        pos_max = _limit_value(limits, "position_max", joint_idx)
        if pos_min is not None:
            lines.append(LimitLine(axis, pos_min, "Position limit", grey))
        if pos_max is not None:
            lines.append(LimitLine(axis, pos_max, "_nolegend_", grey))
    return lines


def limit_color_float(line: LimitLine) -> tuple[float, float, float]:
    return rgb_to_float(line.color)


def finite_concat(arrays: list[np.ndarray]) -> np.ndarray:
    finite_arrays = []
    for values in arrays:
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_arrays.append(arr)
    if not finite_arrays:
        return np.array([])
    return np.concatenate(finite_arrays)


def range_with_limit_values(data_values: list[np.ndarray], limit_values: np.ndarray | None, *, zero_center: bool = False) -> tuple[float, float] | None:
    arrays = list(data_values)
    if limit_values is not None:
        arrays.append(np.asarray(limit_values, dtype=float))
    finite = finite_concat(arrays)
    if finite.size == 0:
        return None
    if zero_center:
        return zero_center_range(finite)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    margin = (hi - lo) * 0.05
    return lo - margin, hi + margin


def align_zero_y_ranges(
    left_range: tuple[float, float],
    right_range: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expand two y-ranges so y=0 lands at the same screen fraction.

    The original bounds are never shrunk. If one side is strictly positive or
    strictly negative, the range is expanded to include zero so the shared
    reference line is visible on both axes.
    """

    def extents(axis_range: tuple[float, float]) -> tuple[float, float]:
        lo, hi = sorted((float(axis_range[0]), float(axis_range[1])))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return 1.0, 1.0
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        return max(-lo, 0.0), max(hi, 0.0)

    left_neg, left_pos = extents(left_range)
    right_neg, right_pos = extents(right_range)
    common_neg = max(left_neg, right_neg)
    common_pos = max(left_pos, right_pos)
    if common_neg <= 0.0 and common_pos <= 0.0:
        return (-1.0, 1.0), (-1.0, 1.0)

    if common_neg <= 0.0:
        zero_fraction = 0.0
    elif common_pos <= 0.0:
        zero_fraction = 1.0
    else:
        zero_fraction = common_neg / (common_neg + common_pos)

    def expand(axis_range: tuple[float, float]) -> tuple[float, float]:
        neg, pos = extents(axis_range)
        if zero_fraction <= 0.0:
            return 0.0, max(pos, 1.0)
        if zero_fraction >= 1.0:
            return -max(neg, 1.0), 0.0
        total = max(neg / zero_fraction, pos / (1.0 - zero_fraction), 1.0)
        return -zero_fraction * total, (1.0 - zero_fraction) * total

    return expand(left_range), expand(right_range)


def axis_assignment_for_slots(slot_states: list[SlotState], view_mode: ViewMode) -> tuple[bool, list[int], list[int]]:
    signals = [state.signal for state in slot_states[:3]]
    active = [i for i in enabled_slot_indices(view_mode) if signals[i] != "None"]
    if view_mode in {"frequency", "scatter"} or len(active) <= 1:
        return False, active, []
    by_unit: dict[str, list[int]] = {}
    units_in_order: list[str] = []
    for idx in active:
        unit = signal_unit(signals[idx])
        if unit not in by_unit:
            by_unit[unit] = []
            units_in_order.append(unit)
        by_unit[unit].append(idx)
    if len(units_in_order) == 1:
        return False, active, []
    if len(units_in_order) == 2:
        first = by_unit[units_in_order[0]]
        second = by_unit[units_in_order[1]]
        left, right = (second, first) if len(second) > len(first) else (first, second)
        return True, sorted(left), sorted(right)
    ordered = sorted(active)
    return True, ordered[:2], ordered[2:]


def series_for_slot(signal_fields: dict[str, np.ndarray], slot_state: SlotState, joint_idx: int, fs: float, fft_cfg: sp.FFTCfg, view_mode: ViewMode, times: np.ndarray | None = None, time_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    source = signal_fields.get(slot_state.signal)
    if source is None or joint_idx >= source.shape[1]:
        return np.array([]), np.array([])
    raw = np.asarray(source[:, joint_idx], dtype=float)
    filtered = sp.apply_filter(raw, fs=fs, cfg=slot_state.filter_cfg)
    if view_mode == "frequency":
        if times is not None and time_range is not None:
            _, filtered = apply_time_range(times, filtered, time_range)
        return sp.compute_spectrum(filtered, fs=fs, cfg=fft_cfg)
    return np.array([]), filtered


def scatter_xy_for_slots(signal_fields: dict[str, np.ndarray], slot_states: list[SlotState], joint_idx: int, fs: float, times: np.ndarray | None = None, time_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if len(slot_states) < 2:
        return np.array([]), np.array([])
    x_state, y_state = slot_states[0], slot_states[1]
    if x_state.signal == "None" or y_state.signal == "None":
        return np.array([]), np.array([])
    x_source = signal_fields.get(x_state.signal)
    y_source = signal_fields.get(y_state.signal)
    if x_source is None or y_source is None:
        return np.array([]), np.array([])
    if joint_idx >= x_source.shape[1] or joint_idx >= y_source.shape[1]:
        return np.array([]), np.array([])
    x = sp.apply_filter(np.asarray(x_source[:, joint_idx], dtype=float), fs=fs, cfg=x_state.filter_cfg)
    y = sp.apply_filter(np.asarray(y_source[:, joint_idx], dtype=float), fs=fs, cfg=y_state.filter_cfg)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    if times is not None and time_range is not None:
        t = np.asarray(times, dtype=float)[:n]
        start, end = time_range
        if end < start:
            start, end = end, start
        in_range = (t >= float(start)) & (t <= float(end))
        x = x[in_range]
        y = y[in_range]
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def sample_overlay_color(base_color: tuple[int, int, int], override_color: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if override_color is not None:
        return override_color
    return tuple(255 - max(0, min(255, int(channel))) for channel in base_color)


def sample_hz_label(sample_hz: float) -> str:
    value = float(sample_hz)
    if value.is_integer():
        return f"{int(value)}Hz"
    return f"{value:g}Hz"


def sample_legend_label(base_label: str, sample_hz: float) -> str:
    return f"{base_label} ({sample_hz_label(sample_hz)} sampled)"


def axis_label_for_slot_states(slot_states: list[SlotState], slot_indices: list[int]) -> str:
    units: list[str] = []
    for idx in slot_indices:
        unit = signal_unit(slot_states[idx].signal)
        if unit and unit not in units:
            units.append(unit)
    return " / ".join(units) if units else ""


def fft_y_label(fft_cfg: sp.FFTCfg) -> str:
    if fft_cfg.scale == "db":
        return "dB"
    return "PSD" if fft_cfg.psd else "magnitude"


def matplotlib_line_style(style: str) -> str:
    from .catalog import LINE_STYLE_LABEL_BY_STYLE
    return style if style in LINE_STYLE_LABEL_BY_STYLE else "-"


def export_figure_size_and_dpi(n_rows: int, n_cols: int, pixel_size: tuple[int, int] | None, dpi: int) -> tuple[tuple[float, float], int]:
    if pixel_size is None:
        return (4.6 * n_cols, 2.8 * n_rows), dpi
    width_px, height_px = pixel_size
    export_dpi = 120
    return (max(1, int(width_px)) / export_dpi, max(1, int(height_px)) / export_dpi), export_dpi
