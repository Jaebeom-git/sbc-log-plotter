# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""Matplotlib PNG export for log_plot.

This module has no Qt dependency and is the common renderer used by GUI
"Save as PNG" and YAML save mode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import model
from . import signal_pipeline as sp
from .catalog import SIGNAL_LABEL_BY_FIELD
from .plotting import (
    apply_time_range,
    align_zero_y_ranges,
    axis_assignment_for_slots,
    axis_label_for_slot_states,
    export_figure_size_and_dpi,
    fft_y_label,
    joint_title,
    limit_color_float,
    limit_lines_for_signal,
    matplotlib_line_style,
    plot_cells_for_group,
    plot_grid_shape_for_cells,
    plot_grid_shape,
    range_with_limit_values,
    rgb_to_float,
    sample_legend_label,
    sample_overlay_color,
    sample_xy_for_hz,
    scatter_xy_for_slots,
    series_for_slot,
    slot_color_float,
    slot_color_rgb,
)
from .state import SlotState, ViewMode


def _plot_matplotlib_time_or_frequency(
    ax,
    *,
    data: model.SbcLogData,
    signal_fields: dict[str, np.ndarray],
    slot_states: list[SlotState],
    joint_idx: int,
    view_mode: ViewMode,
    fs: float,
    fft_cfg: sp.FFTCfg,
    right_slots: list[int],
    left_slots: list[int],
    freq_range: tuple[float, float] | None,
    time_range: tuple[float, float] | None,
    show_limits: bool,
    limits: dict[str, np.ndarray],
    sample_hz: float | None = None,
    sample_color: tuple[int, int, int] | None = None,
):
    right_ax = ax.twinx() if right_slots else None
    legend_handles = []
    legend_labels = []
    left_arrays: list[np.ndarray] = []
    right_arrays: list[np.ndarray] = []
    left_limit_values: list[float] = []
    right_limit_values: list[float] = []
    for side, axis, slot_indices in (("left", ax, left_slots), ("right", right_ax, right_slots)):
        if axis is None:
            continue
        data_arrays = left_arrays if side == "left" else right_arrays
        axis_limit_values = left_limit_values if side == "left" else right_limit_values
        for slot_idx in slot_indices:
            state = slot_states[slot_idx]
            x, y = series_for_slot(
                signal_fields,
                state,
                joint_idx,
                fs,
                fft_cfg,
                view_mode,
                times=data.time,
                time_range=time_range,
            )
            if view_mode == "time":
                x, y = apply_time_range(data.time, y, time_range)
            if x.size == 0 or y.size == 0:
                continue
            label = SIGNAL_LABEL_BY_FIELD.get(state.signal, state.signal)
            (line,) = axis.plot(
                x,
                y,
                color=slot_color_float(state),
                linestyle=matplotlib_line_style(state.line_style),
                linewidth=1.2,
                label=label,
            )
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
            if view_mode == "time" and sample_hz is not None and sample_hz > 0.0:
                sample_x, sample_y = sample_xy_for_hz(x, y, sample_hz)
                if sample_x.size and sample_y.size:
                    overlay_color = sample_overlay_color(slot_color_rgb(state), sample_color)
                    sample_label = sample_legend_label(label, sample_hz)
                    (sample_line,) = axis.plot(
                        sample_x,
                        sample_y,
                        color=rgb_to_float(overlay_color),
                        linestyle="-",
                        marker="o",
                        markersize=2.2,
                        linewidth=0.9,
                        alpha=0.8,
                        label=sample_label,
                    )
                    if sample_label not in legend_labels:
                        legend_handles.append(sample_line)
                        legend_labels.append(sample_label)
            data_arrays.append(y)
            if view_mode == "time" and show_limits:
                for limit_line in limit_lines_for_signal(state.signal, joint_idx, limits, axis="y"):
                    axis_limit_values.append(limit_line.value)
                    handle = axis.axhline(
                        limit_line.value,
                        color=limit_color_float(limit_line),
                        linestyle=limit_line.style,
                        linewidth=0.9,
                        label=limit_line.label,
                    )
                    if limit_line.label != "_nolegend_" and limit_line.label not in legend_labels:
                        legend_handles.append(handle)
                        legend_labels.append(limit_line.label)
    if view_mode == "frequency":
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel(fft_y_label(fft_cfg))
        if freq_range is not None:
            ax.set_xlim(*freq_range)
    else:
        ax.set_xlabel("time [s]")
        ax.set_ylabel(axis_label_for_slot_states(slot_states, left_slots))
    if right_ax is not None:
        right_ax.set_ylabel(axis_label_for_slot_states(slot_states, right_slots))
        right_ax.grid(False)
        if view_mode == "time":
            left_range = range_with_limit_values(
                left_arrays,
                np.asarray(left_limit_values, dtype=float),
            )
            right_range = range_with_limit_values(
                right_arrays,
                np.asarray(right_limit_values, dtype=float),
            )
            if left_range is not None and right_range is not None:
                left_range, right_range = align_zero_y_ranges(left_range, right_range)
                ax.set_ylim(*left_range)
                right_ax.set_ylim(*right_range)
    ax.grid(True, alpha=0.25)
    return legend_handles, legend_labels


def _plot_matplotlib_scatter(
    ax,
    *,
    data: model.SbcLogData,
    signal_fields: dict[str, np.ndarray],
    slot_states: list[SlotState],
    joint_idx: int,
    fs: float,
    time_range: tuple[float, float] | None,
    zero_center_axes: bool = False,
    show_limits: bool = False,
    limits: dict[str, np.ndarray] | None = None,
):
    x_signal = slot_states[0].signal
    y_signal = slot_states[1].signal
    y_state = slot_states[1]
    x, y = scatter_xy_for_slots(signal_fields, slot_states, joint_idx, fs, times=data.time, time_range=time_range)
    label = "Usage samples"
    scatter = None
    if x.size and y.size:
        scatter = ax.scatter(x, y, s=8, alpha=0.45, linewidths=0.0, color=slot_color_float(y_state), label=label)
    ax.set_xlabel(SIGNAL_LABEL_BY_FIELD.get(x_signal, x_signal))
    ax.set_ylabel(SIGNAL_LABEL_BY_FIELD.get(y_signal, y_signal))
    limit_handles = []
    limit_labels = []
    x_limit_values: list[float] = []
    y_limit_values: list[float] = []
    if show_limits and limits is not None:
        for line in [
            *limit_lines_for_signal(x_signal, joint_idx, limits, axis="x"),
            *limit_lines_for_signal(y_signal, joint_idx, limits, axis="y"),
        ]:
            if line.axis == "x":
                x_limit_values.append(line.value)
                handle = ax.axvline(line.value, color=limit_color_float(line), linestyle=line.style, linewidth=0.9, label=line.label)
            else:
                y_limit_values.append(line.value)
                handle = ax.axhline(line.value, color=limit_color_float(line), linestyle=line.style, linewidth=0.9, label=line.label)
            if line.label != "_nolegend_" and line.label not in limit_labels:
                limit_handles.append(handle)
                limit_labels.append(line.label)
    if zero_center_axes:
        x_range = range_with_limit_values([x], np.asarray(x_limit_values, dtype=float), zero_center=True)
        y_range = range_with_limit_values([y], np.asarray(y_limit_values, dtype=float), zero_center=True)
        if x_range is not None:
            ax.set_xlim(*x_range)
        if y_range is not None:
            ax.set_ylim(*y_range)
    elif show_limits:
        x_range = range_with_limit_values([x], np.asarray(x_limit_values, dtype=float))
        y_range = range_with_limit_values([y], np.asarray(y_limit_values, dtype=float))
        if x_range is not None:
            ax.set_xlim(*x_range)
        if y_range is not None:
            ax.set_ylim(*y_range)
    ax.grid(True, alpha=0.25)
    handles = [scatter] if scatter is not None else []
    labels = [label] if scatter is not None else []
    handles.extend(limit_handles)
    labels.extend(limit_labels)
    return handles, labels


def export_current_view_png(
    output_path: str | Path,
    *,
    data: model.SbcLogData,
    signal_fields: dict[str, np.ndarray],
    slot_states: list[SlotState],
    lower_body_names: list[str],
    view_mode: ViewMode,
    fs: float,
    fft_cfg: sp.FFTCfg,
    title: str,
    freq_range: tuple[float, float] | None = None,
    time_range: tuple[float, float] | None = None,
    tile_cols: int = 3,
    pixel_size: tuple[int, int] | None = None,
    scatter_zero_center_axes: bool = False,
    show_limits: bool = False,
    sample_hz: float | None = None,
    sample_color: tuple[int, int, int] | None = None,
    dpi: int = 300,
    plot_group: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names_to_draw = [name for name in lower_body_names if name in data.joint_names]
    if not names_to_draw:
        raise ValueError("No drawable joints found")

    cells = (
        plot_cells_for_group(plot_group, names_to_draw)
        if plot_group is not None
        else []
    )
    n_rows, n_cols = (
        plot_grid_shape_for_cells(cells)
        if cells
        else plot_grid_shape(len(names_to_draw), tile_cols=tile_cols)
    )
    figsize, export_dpi = export_figure_size_and_dpi(n_rows, n_cols, pixel_size, dpi)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=view_mode != "scatter", sharey=False)
    axes_arr = np.asarray(axes, dtype=object).reshape(n_rows, n_cols)
    for ax in axes_arr.reshape(-1):
        ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=14)
    _, left_slots, right_slots = axis_assignment_for_slots(slot_states, view_mode)
    legend_handles = []
    legend_labels = []

    cells_to_draw = cells or [
        type("_Cell", (), {"joint_name": name, "row": idx // n_cols, "col": idx % n_cols})()
        for idx, name in enumerate(names_to_draw)
    ]
    for cell in cells_to_draw:
        joint_name = cell.joint_name
        row, col = cell.row, cell.col
        ax = axes_arr[row, col]
        ax.axis("on")
        joint_idx = data.joint_names.index(joint_name)
        ax.set_title(joint_title(joint_name), fontsize=9)
        if view_mode == "scatter":
            handles, labels = _plot_matplotlib_scatter(
                ax,
                data=data,
                signal_fields=signal_fields,
                slot_states=slot_states,
                joint_idx=joint_idx,
                fs=fs,
                time_range=time_range,
                zero_center_axes=scatter_zero_center_axes,
                show_limits=show_limits,
                limits=data.limits,
            )
        else:
            handles, labels = _plot_matplotlib_time_or_frequency(
                ax,
                data=data,
                signal_fields=signal_fields,
                slot_states=slot_states,
                joint_idx=joint_idx,
                view_mode=view_mode,
                fs=fs,
                fft_cfg=fft_cfg,
                right_slots=right_slots,
                left_slots=left_slots,
                freq_range=freq_range,
                time_range=time_range,
                show_limits=show_limits,
                limits=data.limits,
                sample_hz=sample_hz,
                sample_color=sample_color,
            )
        for handle, label in zip(handles, labels, strict=False):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=min(3, len(legend_labels)), fontsize=8)
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.95))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=export_dpi, bbox_inches="tight")
    plt.close(fig)
    return output
