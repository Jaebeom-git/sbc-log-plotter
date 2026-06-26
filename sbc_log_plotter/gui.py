#!/usr/bin/env python3
# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""Standalone GUI for SBC text logs."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from . import catalog, plotting, state
from . import export as png_export
from . import model, sbc_log_io
from . import signal_pipeline as sp


# Re-export GUI-facing catalog/state/plotting names for tests and call sites.
DERIVED_MOTOR_FIELDS = catalog.DERIVED_MOTOR_FIELDS
SIGNAL_LABEL_BY_FIELD = catalog.SIGNAL_LABEL_BY_FIELD
SIGNAL_FIELD_BY_LABEL = catalog.SIGNAL_FIELD_BY_LABEL
SIGNAL_COLORS = catalog.SIGNAL_COLORS
ALL_SIGNAL_FIELDS = catalog.ALL_SIGNAL_FIELDS
SLOT_OPTIONS = catalog.SLOT_OPTIONS
SIGNAL_UNIT_BY_FIELD = catalog.SIGNAL_UNIT_BY_FIELD
FILTER_CHOICES = catalog.FILTER_CHOICES
FILTER_LABEL_TO_KIND = catalog.FILTER_LABEL_TO_KIND
FILTER_KIND_TO_LABEL = catalog.FILTER_KIND_TO_LABEL
PHASE_CHOICES = catalog.PHASE_CHOICES
PHASE_LABEL_TO_VALUE = catalog.PHASE_LABEL_TO_VALUE
PHASE_VALUE_TO_LABEL = catalog.PHASE_VALUE_TO_LABEL
FFT_WINDOW_OPTIONS = catalog.FFT_WINDOW_OPTIONS
FFT_SCALE_OPTIONS = catalog.FFT_SCALE_OPTIONS
FFT_DETREND_OPTIONS = catalog.FFT_DETREND_OPTIONS
LINE_STYLE_CHOICES = catalog.LINE_STYLE_CHOICES
LINE_STYLE_BY_LABEL = catalog.LINE_STYLE_BY_LABEL
LINE_STYLE_LABEL_BY_STYLE = catalog.LINE_STYLE_LABEL_BY_STYLE
COLOR_CHOICES = catalog.COLOR_CHOICES
COLOR_BY_LABEL = catalog.COLOR_BY_LABEL
SAMPLE_COLOR_CHOICES = catalog.SAMPLE_COLOR_CHOICES
SAMPLE_COLOR_BY_LABEL = catalog.SAMPLE_COLOR_BY_LABEL
PNG_SIZE_CHOICES = catalog.PNG_SIZE_CHOICES
PNG_SIZE_BY_LABEL = catalog.PNG_SIZE_BY_LABEL

ViewMode = state.ViewMode
InputMode = state.InputMode
SlotState = state.SlotState
_LimitLine = plotting.LimitLine

_signal_unit = plotting.signal_unit
build_signal_fields = plotting.build_signal_fields
_field_has_available_samples = plotting.field_has_available_samples
_slot_options_for_signal_fields = plotting.slot_options_for_signal_fields
_enabled_slot_indices = plotting.enabled_slot_indices
_matplotlib_line_style = plotting.matplotlib_line_style
_scatter_xy_for_slots = plotting.scatter_xy_for_slots
_slot_color_rgb = plotting.slot_color_rgb
_rgb_to_float = plotting.rgb_to_float
_slot_color_float = plotting.slot_color_float
_sample_overlay_color = plotting.sample_overlay_color
_sample_hz_label = plotting.sample_hz_label
_sample_legend_label = plotting.sample_legend_label
_joint_title = plotting.joint_title
_lower_body_names_from_joint_names = plotting.lower_body_names_from_joint_names
_plot_groups_from_joint_names = plotting.plot_groups_from_joint_names
_plot_groups_from_layout = plotting.plot_groups_from_layout
_plot_cells_for_group = plotting.plot_cells_for_group
_plot_grid_shape_for_cells = plotting.plot_grid_shape_for_cells
_plot_grid_shape = plotting.plot_grid_shape
_export_figure_size_and_dpi = plotting.export_figure_size_and_dpi
_apply_time_range = plotting.apply_time_range
_sample_indices_for_hz = plotting.sample_indices_for_hz
_sample_xy_for_hz = plotting.sample_xy_for_hz
_zero_center_range = plotting.zero_center_range
_limit_lines_for_signal = plotting.limit_lines_for_signal
_limit_color_float = plotting.limit_color_float
_finite_concat = plotting.finite_concat
_range_with_limit_values = plotting.range_with_limit_values
_align_zero_y_ranges = plotting.align_zero_y_ranges
_axis_assignment_for_slots = plotting.axis_assignment_for_slots
_series_for_slot = plotting.series_for_slot
_axis_label_for_slot_states = plotting.axis_label_for_slot_states
_fft_y_label = plotting.fft_y_label
_plot_matplotlib_time_or_frequency = png_export._plot_matplotlib_time_or_frequency
_plot_matplotlib_scatter = png_export._plot_matplotlib_scatter
export_current_view_png = png_export.export_current_view_png


def _same_unit(a: str, b: str) -> bool:
    ua, ub = _signal_unit(a), _signal_unit(b)
    return a != "None" and b != "None" and ua != "" and ua == ub


def _resolve_png_size(label: str, custom_width: int, custom_height: int) -> tuple[int, int]:
    preset = PNG_SIZE_BY_LABEL.get(label)
    if preset is not None:
        return preset
    return max(1, int(custom_width)), max(1, int(custom_height))


def _limit_option_visible(view_mode: ViewMode) -> bool:
    return view_mode in {"time", "scatter"}


def _robot_limits_available(robot: str = "") -> bool:
    return False


APP_STYLESHEET = """
QWidget#mainPanel {
    background: #0b1020;
    font-size: 12px;
}
QWidget#mainPanel QLineEdit,
QWidget#mainPanel QSpinBox,
QWidget#mainPanel QDoubleSpinBox {
    padding: 3px 6px;
    border: 1px solid #4a5568;
    border-radius: 5px;
    background: #111827;
    color: #f9fafb;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QWidget#mainPanel QComboBox {
    padding: 3px 6px;
    border: 1px solid #4a5568;
    border-radius: 5px;
    background: #ffffff;
    color: #111827;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QWidget#mainPanel QComboBox QAbstractItemView {
    border: 1px solid #4a5568;
    background: #ffffff;
    color: #111827;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QWidget#mainPanel QComboBox:disabled,
QWidget#mainPanel QLineEdit:disabled,
QWidget#mainPanel QSpinBox:disabled,
QWidget#mainPanel QDoubleSpinBox:disabled {
    background: #e5e7eb;
    color: #6b7280;
}
QWidget#mainPanel QPushButton {
    padding: 5px 12px;
    border: 1px solid #4b5563;
    border-radius: 6px;
    background: #2563eb;
    color: white;
    font-weight: 600;
}
QWidget#mainPanel QPushButton:hover {
    background: #1d4ed8;
}
QFrame#controlPanel, QFrame#exportPanel {
    border: 1px solid #1f2937;
    border-radius: 10px;
    background: #111827;
}
QWidget#mainPanel QRadioButton,
QWidget#mainPanel QCheckBox,
QWidget#mainPanel QLabel {
    color: #e5e7eb;
}
"""

VIEW_TIME_LABEL = "Time domain"
VIEW_SCATTER_LABEL = "Scatter"
VIEW_FREQ_LABEL = "Frequency domain"
PLOT_OPTION_GRID_ORDER = ["limit_check", "scatter_zero_center_check"]
PLOT_GROUP_ORDER = ["Lower Body", "Upper Body", "Waist", "All"]


def _selected_log_path(args: argparse.Namespace) -> str:
    positional = str(getattr(args, "path", "") or "")
    if positional:
        return positional
    return str(getattr(args, "sbc_log", "") or "")


@dataclass
class _PlotBundle:
    plot: object
    joint_idx: int
    left_curves: list  # list[(slot_idx, curve)]
    right_curves: list  # list[(slot_idx, curve)]
    left_sample_items: list = field(default_factory=list)  # list[(slot_idx, item)]
    right_sample_items: list = field(default_factory=list)  # list[(slot_idx, item)]
    right_view: object | None = None
    resize_callback: object | None = None
    scatter_item: object | None = None
    limit_items: list = field(default_factory=list)


def _require_gui():
    try:
        from PySide6 import QtCore, QtWidgets
        import pyqtgraph as pg
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PySide6/pyqtgraph가 설치되어 있지 않습니다. `pixi install` 후 다시 실행하세요."
        ) from exc
    return QtCore, QtWidgets, pg


def _make_clickable_view_box(pg, owner: "SbcLogPlotWindow", joint_name: str):
    class ClickableViewBox(pg.ViewBox):
        def __init__(self_inner):
            super().__init__()
            # X-axis only zoom/pan via mouse
            self_inner.setMouseEnabled(x=True, y=False)

        def mouseDoubleClickEvent(self_inner, event):  # noqa: N802
            owner.toggle_focus(joint_name)
            event.accept()

        def wheelEvent(self_inner, event, axis=None):  # noqa: N802
            # Force zoom on X axis regardless of cursor position
            super().wheelEvent(event, axis=0)

    return ClickableViewBox()


class SbcLogPlotWindow:
    def __init__(self, args: argparse.Namespace) -> None:
        QtCore, QtWidgets, pg = _require_gui()
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.pg = pg
        pg.setConfigOptions(antialias=False, background="k", foreground="w")
        self.args = args
        self.mode: InputMode = "sbc"
        self.data: model.SbcLogData | None = None
        self.signal_fields: dict[str, np.ndarray] = {}
        self.motor_layout: model.MotorLayout = sbc_log_io.default_motor_layout()
        self.lower_body_names: list[str] = self._resolve_lower_body_names(self.motor_layout)
        self.plot_groups: dict[str, list[str]] = _plot_groups_from_layout(self.motor_layout)
        self.current_plot_group: str = "Lower Body"
        self.fs: float = float(args.sample_rate_hz)
        self.view_mode: ViewMode = "time"
        self.slot_states: list[SlotState] = [SlotState() for _ in range(3)]
        self.fft_cfg = sp.FFTCfg(scale="linear", psd=True)
        self.plot_bundles: list[_PlotBundle] = []
        self.focused_joint: str | None = None
        self.robot_limits_available = False

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("SBC Log Plot")
        self.window.resize(1800, 1100)
        central = QtWidgets.QWidget()
        central.setObjectName("mainPanel")
        self.window.setCentralWidget(central)
        self._apply_style()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self._build_controls(root)

        self.graphics = pg.GraphicsLayoutWidget()
        root.addWidget(self.graphics, stretch=1)
        self.legend_label = QtWidgets.QLabel()
        self.legend_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.legend_label.setWordWrap(True)
        root.addWidget(self.legend_label)
        export_panel = QtWidgets.QFrame()
        export_panel.setObjectName("exportPanel")
        bottom = QtWidgets.QHBoxLayout(export_panel)
        bottom.setContentsMargins(10, 8, 10, 8)
        bottom.setSpacing(8)
        self.status_label = QtWidgets.QLabel("Ready")
        self.save_title_label = QtWidgets.QLabel("Title")
        self.save_title_edit = QtWidgets.QLineEdit()
        self.save_title_edit.setPlaceholderText("Use default title")
        self.save_title_edit.setMaximumWidth(280)
        self.save_size_label = QtWidgets.QLabel("PNG size")
        self.save_size_combo = QtWidgets.QComboBox()
        self.save_size_combo.addItems([label for label, _ in PNG_SIZE_CHOICES])
        self.save_width_spin = QtWidgets.QSpinBox()
        self.save_width_spin.setRange(320, 10000)
        self.save_width_spin.setValue(2048)
        self.save_height_spin = QtWidgets.QSpinBox()
        self.save_height_spin.setRange(240, 10000)
        self.save_height_spin.setValue(1536)
        self.save_size_x_label = QtWidgets.QLabel("×")
        self.save_png_button = QtWidgets.QPushButton("Save as PNG")
        self.save_png_button.clicked.connect(self._save_as_png)
        self.save_size_combo.currentTextChanged.connect(self._on_save_size_changed)
        bottom.addWidget(self.status_label, stretch=1)
        bottom.addWidget(self.save_title_label)
        bottom.addWidget(self.save_title_edit)
        bottom.addWidget(self.save_size_label)
        bottom.addWidget(self.save_size_combo)
        bottom.addWidget(self.save_width_spin)
        bottom.addWidget(self.save_size_x_label)
        bottom.addWidget(self.save_height_spin)
        bottom.addWidget(self.save_png_button)
        root.addWidget(export_panel)
        self._on_save_size_changed(self.save_size_combo.currentText())
        self._sync_controls_for_view_mode()
        self._update_legend()

    def _apply_style(self) -> None:
        self.window.setStyleSheet(APP_STYLESHEET)

    @staticmethod
    def _resolve_lower_body_names(layout: model.MotorLayout) -> list[str]:
        lower_indices = layout.groups.get("Lower Body", [])
        return [layout.names[i] for i in lower_indices]

    def _available_plot_group_names(self) -> list[str]:
        ordered = [name for name in PLOT_GROUP_ORDER if self.plot_groups.get(name)]
        extras = [name for name in self.plot_groups if name not in PLOT_GROUP_ORDER and self.plot_groups.get(name)]
        return [*ordered, *extras]

    def _current_group_names(self) -> list[str]:
        names = self.plot_groups.get(self.current_plot_group)
        if names:
            return names
        return self.plot_groups.get("Lower Body", self.lower_body_names)

    def _refresh_plot_group_options(self) -> None:
        if not hasattr(self, "plot_group_combo"):
            return
        options = self._available_plot_group_names() or ["Lower Body"]
        current = self.current_plot_group if self.current_plot_group in options else options[0]
        self.current_plot_group = current
        self.plot_group_combo.blockSignals(True)
        self.plot_group_combo.clear()
        self.plot_group_combo.addItems(options)
        self.plot_group_combo.setCurrentText(current)
        self.plot_group_combo.blockSignals(False)

    def _build_controls(self, root) -> None:
        QtW = self.QtWidgets
        controls_panel = QtW.QFrame()
        controls_panel.setObjectName("controlPanel")
        grid = QtW.QGridLayout(controls_panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        root.addWidget(controls_panel)

        # Row 0: plot group selector
        self.plot_group_combo = QtW.QComboBox()
        self.plot_group_combo.addItems(self._available_plot_group_names() or ["Lower Body"])
        self.plot_group_combo.setCurrentText(self.current_plot_group)
        self.plot_group_combo.currentTextChanged.connect(self._on_plot_group_changed)
        grid.addWidget(QtW.QLabel("Plot group"), 0, 0)
        grid.addWidget(self.plot_group_combo, 0, 1)

        # Row 1: source-specific layout selector
        self.yaml_edit = QtW.QLineEdit(self.args.yaml or "")
        yaml_browse = QtW.QPushButton("Browse")
        grid.addWidget(QtW.QLabel("YAML config"), 1, 0)
        grid.addWidget(self.yaml_edit, 1, 1, 1, 6)
        grid.addWidget(yaml_browse, 1, 7)
        yaml_browse.clicked.connect(self._browse_yaml)

        # Row 2: log file
        initial_log = _selected_log_path(self.args)
        self.sbc_file_edit = QtW.QLineEdit(initial_log or "")
        sbc_browse = QtW.QPushButton("Browse")
        load = QtW.QPushButton("Load / Refresh")
        log_label = "SBC log file"
        grid.addWidget(QtW.QLabel(log_label), 2, 0)
        grid.addWidget(self.sbc_file_edit, 2, 1, 1, 6)
        grid.addWidget(sbc_browse, 2, 7)
        grid.addWidget(load, 2, 8)
        sbc_browse.clicked.connect(self._browse_sbc_file)
        load.clicked.connect(self.reload_data)

        # Row 3: sample rate
        self.sample_rate_spin = QtW.QDoubleSpinBox()
        self.sample_rate_spin.setRange(1.0, 100000.0)
        self.sample_rate_spin.setValue(float(self.args.sample_rate_hz))
        grid.addWidget(QtW.QLabel("Sample Rate [Hz]"), 3, 0)
        grid.addWidget(self.sample_rate_spin, 3, 1)
        self.sample_marker_check = QtW.QCheckBox("Sample markers")
        self.sample_marker_check.setChecked(False)
        self.sample_marker_hz_spin = QtW.QDoubleSpinBox()
        self.sample_marker_hz_spin.setRange(0.1, 100000.0)
        self.sample_marker_hz_spin.setDecimals(1)
        self.sample_marker_hz_spin.setValue(50.0)
        self.sample_marker_hz_spin.setSuffix(" Hz")
        self.sample_marker_color_combo = QtW.QComboBox()
        self.sample_marker_color_combo.addItems([label for label, _ in SAMPLE_COLOR_CHOICES])
        self.sample_marker_color_combo.setCurrentText("Complement")
        grid.addWidget(self.sample_marker_check, 3, 2)
        grid.addWidget(self.sample_marker_hz_spin, 3, 3)
        grid.addWidget(self.sample_marker_color_combo, 3, 4)
        self.sample_marker_widgets = [
            self.sample_marker_check,
            self.sample_marker_hz_spin,
            self.sample_marker_color_combo,
        ]

        # Row 4: view mode + time range (applies to time/scatter/frequency)
        self.view_time_radio = QtW.QRadioButton(VIEW_TIME_LABEL)
        self.view_scatter_radio = QtW.QRadioButton(VIEW_SCATTER_LABEL)
        self.view_freq_radio = QtW.QRadioButton(VIEW_FREQ_LABEL)
        self.view_time_radio.setChecked(True)
        grid.addWidget(QtW.QLabel("View"), 4, 0)
        grid.addWidget(self.view_time_radio, 4, 1)
        grid.addWidget(self.view_scatter_radio, 4, 2)
        grid.addWidget(self.view_freq_radio, 4, 3)

        self.time_min_spin = QtW.QDoubleSpinBox()
        self.time_min_spin.setRange(-1e9, 1e9)
        self.time_min_spin.setDecimals(3)
        self.time_min_spin.setValue(0.0)
        self.time_max_spin = QtW.QDoubleSpinBox()
        self.time_max_spin.setRange(-1e9, 1e9)
        self.time_max_spin.setDecimals(3)
        self.time_max_spin.setValue(10.0)
        self.time_range_label = QtW.QLabel("Time range [s]")
        self.time_range_dash = QtW.QLabel("-")
        grid.addWidget(self.time_range_label, 4, 4)
        grid.addWidget(self.time_min_spin, 4, 5)
        grid.addWidget(self.time_range_dash, 4, 6)
        grid.addWidget(self.time_max_spin, 4, 7)

        self.time_range_widgets = [
            self.time_range_label,
            self.time_min_spin,
            self.time_range_dash,
            self.time_max_spin,
        ]
        self.limit_check = QtW.QCheckBox("Show limits")
        self.limit_check.setChecked(False)
        self.limit_robot_label = QtW.QLabel("Limit robot")
        self.limit_robot_combo = QtW.QComboBox()
        self.limit_robot_combo.addItems(["NONE"])
        self.limit_option_widgets = [self.limit_check]
        self.limit_robot_widgets = [self.limit_robot_label, self.limit_robot_combo]

        # Row 5: frequency range + FFT options (visible in frequency mode)
        self.fmin_spin = QtW.QDoubleSpinBox()
        self.fmin_spin.setRange(0.0, 1e6); self.fmin_spin.setValue(0.0)
        self.fmax_spin = QtW.QDoubleSpinBox()
        self.fmax_spin.setRange(0.1, 1e6); self.fmax_spin.setValue(50.0)
        self.freq_range_label = QtW.QLabel("Freq range [Hz]")
        self.freq_range_dash = QtW.QLabel("-")
        grid.addWidget(self.freq_range_label, 5, 0)
        grid.addWidget(self.fmin_spin, 5, 1)
        grid.addWidget(self.freq_range_dash, 5, 2)
        grid.addWidget(self.fmax_spin, 5, 3)
        self.freq_range_widgets = [
            self.freq_range_label,
            self.fmin_spin,
            self.freq_range_dash,
            self.fmax_spin,
        ]

        self.scatter_options_label = QtW.QLabel("Plot options")
        self.scatter_zero_center_check = QtW.QCheckBox("0-centered symmetric axes")
        self.scatter_zero_center_check.setChecked(False)
        grid.addWidget(self.scatter_options_label, 5, 0)
        grid.addWidget(self.limit_check, 5, 1)
        grid.addWidget(self.scatter_zero_center_check, 5, 2, 1, 3)
        grid.addWidget(self.limit_robot_label, 5, 5)
        grid.addWidget(self.limit_robot_combo, 5, 6)
        self.scatter_option_widgets = [
            self.scatter_zero_center_check,
        ]

        self.fft_window_combo = QtW.QComboBox()
        self.fft_window_combo.addItems(FFT_WINDOW_OPTIONS)
        self.fft_window_combo.setCurrentText("hann")
        self.fft_scale_combo = QtW.QComboBox()
        self.fft_scale_combo.addItems(FFT_SCALE_OPTIONS)
        self.fft_scale_combo.setCurrentText("linear")
        self.fft_detrend_combo = QtW.QComboBox()
        self.fft_detrend_combo.addItems(FFT_DETREND_OPTIONS)
        self.fft_detrend_combo.setCurrentText("mean")
        self.fft_psd_check = QtW.QCheckBox("PSD"); self.fft_psd_check.setChecked(True)
        self.fft_nperseg_spin = QtW.QSpinBox()
        self.fft_nperseg_spin.setRange(64, 65536)
        self.fft_nperseg_spin.setValue(2048)
        self.fft_options_label = QtW.QLabel("FFT options")
        self.fft_window_label = QtW.QLabel("Window")
        self.fft_scale_label = QtW.QLabel("Scale")
        self.fft_detrend_label = QtW.QLabel("Detrend")
        self.fft_nperseg_label = QtW.QLabel("nperseg")
        grid.addWidget(self.fft_options_label, 5, 4)
        grid.addWidget(self.fft_window_label, 5, 5)
        grid.addWidget(self.fft_window_combo, 5, 6)
        grid.addWidget(self.fft_scale_label, 5, 7)
        grid.addWidget(self.fft_scale_combo, 5, 8)
        grid.addWidget(self.fft_detrend_label, 5, 9)
        grid.addWidget(self.fft_detrend_combo, 5, 10)
        grid.addWidget(self.fft_psd_check, 5, 11)
        grid.addWidget(self.fft_nperseg_label, 5, 12)
        grid.addWidget(self.fft_nperseg_spin, 5, 13)
        self.fft_option_widgets = [
            self.fft_options_label,
            self.fft_window_label,
            self.fft_window_combo,
            self.fft_scale_label,
            self.fft_scale_combo,
            self.fft_detrend_label,
            self.fft_detrend_combo,
            self.fft_psd_check,
            self.fft_nperseg_label,
            self.fft_nperseg_spin,
        ]

        # Row 6: slot table header
        header_row = 6
        grid.addWidget(QtW.QLabel("Slot"), header_row, 0)
        grid.addWidget(QtW.QLabel("Signal"), header_row, 1)
        grid.addWidget(QtW.QLabel("Line"), header_row, 2)
        grid.addWidget(QtW.QLabel("Color"), header_row, 3)
        grid.addWidget(QtW.QLabel("Filter"), header_row, 4)
        grid.addWidget(QtW.QLabel("Filter parameters"), header_row, 5, 1, 7)

        # Rows 6..8: slot rows
        self.slot_signal_combos: list = []
        self.slot_style_combos: list = []
        self.slot_color_combos: list = []
        self.slot_filter_combos: list = []
        self.slot_param_containers: list = []
        for i in range(3):
            row = header_row + 1 + i
            grid.addWidget(QtW.QLabel(f"Slot {i + 1}"), row, 0)

            sig_combo = QtW.QComboBox(); sig_combo.addItems(SLOT_OPTIONS)
            sig_combo.currentTextChanged.connect(
                lambda label, idx=i: self._on_signal_changed(idx, SIGNAL_FIELD_BY_LABEL[label])
            )
            grid.addWidget(sig_combo, row, 1)
            self.slot_signal_combos.append(sig_combo)

            style_combo = QtW.QComboBox()
            style_combo.addItems([label for label, _ in LINE_STYLE_CHOICES])
            style_combo.currentTextChanged.connect(
                lambda label, idx=i: self._on_line_style_changed(idx, LINE_STYLE_BY_LABEL[label])
            )
            grid.addWidget(style_combo, row, 2)
            self.slot_style_combos.append(style_combo)

            color_combo = QtW.QComboBox()
            color_combo.addItems([label for label, _ in COLOR_CHOICES])
            color_combo.currentTextChanged.connect(
                lambda label, idx=i: self._on_color_changed(idx, COLOR_BY_LABEL[label])
            )
            grid.addWidget(color_combo, row, 3)
            self.slot_color_combos.append(color_combo)

            filt_combo = QtW.QComboBox()
            filt_combo.addItems([label for label, _ in FILTER_CHOICES])
            filt_combo.currentTextChanged.connect(
                lambda text, idx=i: self._on_filter_kind_changed(idx, FILTER_LABEL_TO_KIND[text])
            )
            grid.addWidget(filt_combo, row, 4)
            self.slot_filter_combos.append(filt_combo)

            container = QtW.QWidget()
            container.setLayout(QtW.QHBoxLayout())
            container.layout().setContentsMargins(0, 0, 0, 0)
            grid.addWidget(container, row, 5, 1, 7)
            self.slot_param_containers.append(container)

        for i in range(3):
            self._rebuild_param_container(i)

        self.view_time_radio.toggled.connect(self._on_view_mode_changed)
        self.view_scatter_radio.toggled.connect(self._on_view_mode_changed)
        self.view_freq_radio.toggled.connect(self._on_view_mode_changed)
        self.time_min_spin.valueChanged.connect(self._on_time_range_changed)
        self.time_max_spin.valueChanged.connect(self._on_time_range_changed)
        self.fmin_spin.valueChanged.connect(self._on_freq_range_changed)
        self.fmax_spin.valueChanged.connect(self._on_freq_range_changed)
        self.sample_marker_check.toggled.connect(self._on_sample_marker_changed)
        self.sample_marker_hz_spin.valueChanged.connect(self._on_sample_marker_changed)
        self.sample_marker_color_combo.currentTextChanged.connect(self._on_sample_marker_changed)
        self.limit_check.toggled.connect(self._on_limit_opt_changed)
        self.limit_robot_combo.currentTextChanged.connect(self._on_limit_robot_changed)
        self.scatter_zero_center_check.toggled.connect(self._on_scatter_opt_changed)
        self.fft_window_combo.currentTextChanged.connect(self._on_fft_opt_changed)
        self.fft_scale_combo.currentTextChanged.connect(self._on_fft_opt_changed)
        self.fft_detrend_combo.currentTextChanged.connect(self._on_fft_opt_changed)
        self.fft_psd_check.toggled.connect(self._on_fft_opt_changed)
        self.fft_nperseg_spin.valueChanged.connect(self._on_fft_opt_changed)

    @staticmethod
    def _set_widgets_visible(widgets: list, visible: bool) -> None:
        for widget in widgets:
            widget.setVisible(visible)
            widget.setEnabled(visible)

    def _sync_controls_for_view_mode(self) -> None:
        self._set_widgets_visible(self.time_range_widgets, True)
        limit_visible = (
            self.robot_limits_available
            and _limit_option_visible(self.view_mode)
        )
        option_label_visible = self.view_mode == "scatter" or limit_visible
        self._set_widgets_visible([self.scatter_options_label], option_label_visible)
        self._set_widgets_visible(self.limit_option_widgets, limit_visible)
        self._set_widgets_visible(self.limit_robot_widgets, limit_visible)
        self._set_widgets_visible(self.sample_marker_widgets, self.view_mode == "time")
        self._set_widgets_visible(self.scatter_option_widgets, self.view_mode == "scatter")
        self._set_widgets_visible(self.freq_range_widgets, self.view_mode == "frequency")
        self._set_widgets_visible(self.fft_option_widgets, self.view_mode == "frequency")
        enabled_slots = set(_enabled_slot_indices(self.view_mode))
        for idx in range(3):
            enabled = idx in enabled_slots
            self.slot_signal_combos[idx].setEnabled(enabled)
            self.slot_style_combos[idx].setEnabled(enabled)
            self.slot_color_combos[idx].setEnabled(enabled)
            self.slot_filter_combos[idx].setEnabled(enabled)
            self.slot_param_containers[idx].setEnabled(enabled)
        self._update_legend()

    def _browse_sbc_file(self) -> None:
        start = self.sbc_file_edit.text().strip() or str(Path.cwd())
        start_dir = str(Path(start).parent) if Path(start).is_file() else start
        title = "Select SBC log file"
        filters = "Text files (*.txt);;All files (*)"
        filename, _ = self.QtWidgets.QFileDialog.getOpenFileName(
            self.window, title,
            start_dir, filters,
        )
        if filename:
            self.sbc_file_edit.setText(filename)
            self.reload_data()

    def _browse_yaml(self) -> None:
        start_dir = self.yaml_edit.text().strip() or str(Path.cwd())
        if Path(start_dir).is_file():
            start_dir = str(Path(start_dir).parent)
        filename, _ = self.QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Select policy YAML config",
            start_dir, "YAML files (*.yaml *.yml);;All files (*)",
        )
        if filename:
            self.yaml_edit.setText(filename)
            self._reload_layout_from_yaml()

    def _save_as_png(self) -> None:
        if self.data is None:
            self.status_label.setText("Save failed: load a log first")
            return
        start = Path(self.sbc_file_edit.text().strip() or Path.cwd()).expanduser()
        if start.is_file():
            default_name = start.with_suffix(".png")
        else:
            default_name = start / "sbc_log_plot.png"
        filename, _ = self.QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save current plots as PNG",
            str(default_name),
            "PNG files (*.png);;All files (*)",
        )
        if not filename:
            return
        output = Path(filename).expanduser()
        if output.suffix.lower() != ".png":
            output = output.with_suffix(".png")
        try:
            log_name = start.stem if start.is_file() else start.name
            if not log_name:
                log_name = "SBC Log"
            mode_label = {
                "time": VIEW_TIME_LABEL,
                "frequency": VIEW_FREQ_LABEL,
                "scatter": VIEW_SCATTER_LABEL,
            }[self.view_mode]
            title = self.save_title_edit.text().strip() or f"{log_name} - {mode_label}"
            names_to_draw = [self.focused_joint] if self.focused_joint else self._current_group_names()
            png_export.export_current_view_png(
                output,
                data=self.data,
                signal_fields=self.signal_fields,
                slot_states=self.slot_states,
                lower_body_names=names_to_draw,
                plot_group=self.current_plot_group,
                view_mode=self.view_mode,
                fs=self.fs,
                fft_cfg=self.fft_cfg,
                title=title,
                freq_range=(float(self.fmin_spin.value()), float(self.fmax_spin.value())),
                time_range=self._current_time_range(),
                pixel_size=self._current_png_size(),
                scatter_zero_center_axes=self.scatter_zero_center_check.isChecked(),
                show_limits=(
                    self.robot_limits_available
                    and _limit_option_visible(self.view_mode)
                    and self.limit_check.isChecked()
                ),
                sample_hz=self._current_sample_hz(),
                sample_color=self._current_sample_color(),
                dpi=300,
            )
            self.status_label.setText(f"Saved PNG: {output}")
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Save failed: {exc}")

    def _current_png_size(self) -> tuple[int, int]:
        return _resolve_png_size(
            self.save_size_combo.currentText(),
            int(self.save_width_spin.value()),
            int(self.save_height_spin.value()),
        )

    def _current_time_range(self) -> tuple[float, float] | None:
        start = float(self.time_min_spin.value())
        end = float(self.time_max_spin.value())
        if end <= start:
            return None
        return start, end

    def _current_sample_hz(self) -> float | None:
        if self.view_mode != "time":
            return None
        if not self.sample_marker_check.isChecked():
            return None
        return float(self.sample_marker_hz_spin.value())

    def _current_sample_color(self) -> tuple[int, int, int] | None:
        if self.view_mode != "time":
            return None
        return SAMPLE_COLOR_BY_LABEL.get(self.sample_marker_color_combo.currentText())

    def _reload_layout_from_yaml(self) -> None:
        yaml_text = self.yaml_edit.text().strip()
        if not yaml_text:
            self.motor_layout = sbc_log_io.default_motor_layout()
        else:
            yaml_path = Path(yaml_text).expanduser()
            repo_root = (
                yaml_path.parent.parent if yaml_path.parent.name == "config"
                else yaml_path.parent
            )
            # Pass the selected YAML directly so motor names come from it rather
            # than a conventional repo-relative path (which may not exist).
            self.motor_layout = sbc_log_io.default_motor_layout(repo_root, policy_yaml=yaml_path)
        self.lower_body_names = self._resolve_lower_body_names(self.motor_layout)
        self.plot_groups = _plot_groups_from_layout(self.motor_layout)
        self._refresh_plot_group_options()
        if self.data is not None:
            self.render_plots()

    def reload_data(self) -> None:
        try:
            log_text = self.sbc_file_edit.text().strip()
            if not log_text:
                raise ValueError("Log path is empty")
            log_path = Path(log_text).expanduser()
            self._reload_layout_from_yaml()
            if not log_path.is_file():
                raise FileNotFoundError(f"Not a file: {log_path}")
            self.data = sbc_log_io.load_sbc_log(
                log_path,
                sample_rate_hz=float(self.sample_rate_spin.value()),
                time_offset=float(self.args.time_offset),
                motor_layout=self.motor_layout,
                limit_robot="",
            )
            self.lower_body_names = self._resolve_lower_body_names(self.motor_layout)
            self.plot_groups = _plot_groups_from_layout(self.motor_layout)
            self._refresh_plot_group_options()
            self.signal_fields = build_signal_fields(self.data)
            self._refresh_slot_signal_options()
            self.fs = float(self.data.sample_rate_hz)
            self.sample_rate_spin.blockSignals(True)
            self.sample_rate_spin.setValue(float(self.fs))
            self.sample_rate_spin.blockSignals(False)
            if self.data.time.size:
                self.time_min_spin.blockSignals(True)
                self.time_max_spin.blockSignals(True)
                self.time_min_spin.setValue(float(self.data.time[0]))
                self.time_max_spin.setValue(float(self.data.time[-1]))
                self.time_min_spin.blockSignals(False)
                self.time_max_spin.blockSignals(False)
            nyq = self.fs / 2.0
            self.fmax_spin.setMaximum(nyq)
            if self.fmax_spin.value() > nyq:
                self.fmax_spin.setValue(nyq)
            self.status_label.setText(
                f"Loaded {log_path.name}: {self.data.time.size} samples @ {self.fs:.1f} Hz"
            )
            self.render_plots()
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Load failed: {exc}")

    def _refresh_slot_signal_options(self) -> None:
        if not hasattr(self, "slot_signal_combos"):
            return
        options = (
            _slot_options_for_signal_fields(self.signal_fields)
            if self.signal_fields
            else SLOT_OPTIONS
        )
        option_set = set(options)
        for idx, combo in enumerate(self.slot_signal_combos):
            current_signal = self.slot_states[idx].signal
            current_label = SIGNAL_LABEL_BY_FIELD.get(current_signal, "None")
            if current_label not in option_set:
                current_signal = "None"
                current_label = SIGNAL_LABEL_BY_FIELD["None"]
                self.slot_states[idx].signal = "None"
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(options)
            combo.setCurrentText(current_label)
            combo.blockSignals(False)
        self._update_legend()

    def _slot_pen(self, slot_idx: int, signal: str):
        color = _slot_color_rgb(self.slot_states[slot_idx])
        style_by_text = {
            "-": self.QtCore.Qt.PenStyle.SolidLine,
            "--": self.QtCore.Qt.PenStyle.DashLine,
            "-.": self.QtCore.Qt.PenStyle.DashDotLine,
            ":": self.QtCore.Qt.PenStyle.DotLine,
        }
        return self.pg.mkPen(
            color,
            width=1.8,
            style=style_by_text.get(self.slot_states[slot_idx].line_style, self.QtCore.Qt.PenStyle.SolidLine),
        )

    def _axis_label_for_slots(self, slot_indices: list[int]) -> str:
        if self.view_mode == "frequency":
            return _fft_y_label(self.fft_cfg)
        units: list[str] = []
        for i in slot_indices:
            unit = _signal_unit(self.slot_states[i].signal)
            if unit and unit not in units:
                units.append(unit)
        return " / ".join(units) if units else ""

    def _update_legend(self) -> None:
        if not hasattr(self, "legend_label"):
            return
        line_marks = {"-": "━━", "--": "╌╌", "-.": "━╴", ":": "┈┈"}
        parts: list[str] = []
        enabled_slots = set(_enabled_slot_indices(self.view_mode))
        for i, state in enumerate(self.slot_states):
            slot_tag = f"Slot {i + 1}"
            if i not in enabled_slots:
                parts.append(
                    f"<span style='color:rgb(110,110,110);'>{slot_tag}: disabled in Scatter</span>"
                )
                continue
            if state.signal == "None":
                parts.append(
                    f"<span style='color:rgb(150,150,150);'>{slot_tag}: None</span>"
                )
                continue
            color = _slot_color_rgb(state)
            rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
            signal_label = SIGNAL_LABEL_BY_FIELD.get(state.signal, state.signal)
            filter_label = FILTER_KIND_TO_LABEL.get(state.filter_cfg.kind, state.filter_cfg.kind)
            axis_tag = ""
            if self.view_mode == "scatter":
                axis_tag = " (X)" if i == 0 else " (Y)"
            line_mark = line_marks.get(state.line_style, "━━")
            parts.append(
                f"<span style='color:{rgb}; font-weight:600;'>"
                f"{line_mark} {slot_tag}{axis_tag}: {signal_label}</span>"
                f" <span style='color:rgb(180,180,180);'>[{filter_label}]</span>"
            )
            sample_hz = self._current_sample_hz()
            if sample_hz is not None:
                overlay_color = _sample_overlay_color(color, self._current_sample_color())
                overlay_rgb = f"rgb({overlay_color[0]}, {overlay_color[1]}, {overlay_color[2]})"
                parts.append(
                    f"<span style='color:{overlay_rgb}; font-weight:600;'>"
                    f"✱ {slot_tag}: {_sample_legend_label(signal_label, sample_hz)}</span>"
                )
        self.legend_label.setText(" &nbsp;|&nbsp; ".join(parts))

    def toggle_focus(self, joint_name: str) -> None:
        self.focused_joint = None if self.focused_joint == joint_name else joint_name
        self.render_plots()

    def _make_right_view(self, plot):
        right_view = self.pg.ViewBox()
        right_view.setMouseEnabled(x=False, y=False)
        plot.scene().addItem(right_view)
        plot.showAxis("right")
        plot.getAxis("right").linkToView(right_view)
        plot.getAxis("right").enableAutoSIPrefix(False)
        right_view.setXLink(plot)

        def update_views(*_args, plot_item=plot, rv=right_view):
            rv.setGeometry(plot_item.getViewBox().sceneBoundingRect())
            rv.linkedViewChanged(plot_item.getViewBox(), rv.XAxis)

        update_views()
        plot.getViewBox().sigResized.connect(update_views)
        return right_view, update_views

    def _clear_plot_bundles(self) -> None:
        for bundle in self.plot_bundles:
            self._clear_limit_items(bundle)
            if bundle.right_view is None or bundle.resize_callback is None:
                continue
            try:
                bundle.plot.getViewBox().sigResized.disconnect(bundle.resize_callback)
            except Exception:
                pass
            try:
                bundle.plot.scene().removeItem(bundle.right_view)
            except Exception:
                pass

    def _clear_limit_items(self, bundle: _PlotBundle) -> None:
        for owner, item in getattr(bundle, "limit_items", []):
            try:
                owner.removeItem(item)
            except Exception:
                pass
        bundle.limit_items = []

    def _make_limit_item(self, line: _LimitLine):
        angle = 90 if line.axis == "x" else 0
        return self.pg.InfiniteLine(
            pos=line.value,
            angle=angle,
            movable=False,
            pen=self.pg.mkPen(line.color, width=1.1, style=self.QtCore.Qt.PenStyle.DashLine),
        )

    def _show_limits_enabled(self) -> bool:
        return (
            self.robot_limits_available
            and _limit_option_visible(self.view_mode)
            and self.limit_check.isChecked()
            and self.data is not None
        )

    def _refresh_limit_items(
        self,
        bundle: _PlotBundle,
        left_slots: list[int],
        right_slots: list[int],
    ) -> None:
        self._clear_limit_items(bundle)
        if not self._show_limits_enabled():
            return
        limits = self.data.limits
        if not limits:
            return
        if self.view_mode == "frequency":
            return
        if self.view_mode == "scatter":
            signals = [self.slot_states[0].signal, self.slot_states[1].signal]
            lines = [
                *_limit_lines_for_signal(signals[0], bundle.joint_idx, limits, axis="x"),
                *_limit_lines_for_signal(signals[1], bundle.joint_idx, limits, axis="y"),
            ]
            for line in lines:
                item = self._make_limit_item(line)
                bundle.plot.addItem(item)
                bundle.limit_items.append((bundle.plot, item))
            return
        for slot_idx in left_slots:
            for line in _limit_lines_for_signal(
                self.slot_states[slot_idx].signal,
                bundle.joint_idx,
                limits,
                axis="y",
            ):
                item = self._make_limit_item(line)
                bundle.plot.addItem(item)
                bundle.limit_items.append((bundle.plot, item))
        if bundle.right_view is None:
            return
        for slot_idx in right_slots:
            for line in _limit_lines_for_signal(
                self.slot_states[slot_idx].signal,
                bundle.joint_idx,
                limits,
                axis="y",
            ):
                item = self._make_limit_item(line)
                bundle.right_view.addItem(item)
                bundle.limit_items.append((bundle.right_view, item))

    def _limit_values_for_signals(
        self,
        signals: list[str],
        joint_idx: int,
        *,
        axis: Literal["x", "y"] = "y",
    ) -> np.ndarray:
        if not self._show_limits_enabled() or self.data is None:
            return np.array([])
        values = [
            line.value
            for signal in signals
            for line in _limit_lines_for_signal(signal, joint_idx, self.data.limits, axis=axis)
            if line.axis == axis
        ]
        return np.asarray(values, dtype=float)

    def _axis_assignment(self) -> tuple[bool, list[int], list[int]]:
        """Decide left/right slot grouping by data count and unit pattern.

        Returns (has_right_axis, left_slot_indices, right_slot_indices).

        Rules (active = slots with a signal selected):
          - 1 active:                          left only.
          - 2 active, same unit:               both left.
          - 2 active, different units:         left / right.
          - 3 active, all same unit:           all left.
          - 3 active, two share a unit:        the matching pair left, odd one right.
          - 3 active, all different units:     two left, one right (axis cap).
        Frequency mode always collapses to one (left) axis.
        """
        signals = [self.slot_states[i].signal for i in range(3)]
        active = [i for i in range(3) if signals[i] != "None"]
        if self.view_mode == "scatter":
            return False, _enabled_slot_indices("scatter"), []
        if self.view_mode == "frequency" or len(active) <= 1:
            return False, active, []

        # Group active slots by unit, preserving slot order.
        by_unit: dict[str, list[int]] = {}
        units_in_order: list[str] = []
        for i in active:
            unit = _signal_unit(signals[i])
            if unit not in by_unit:
                by_unit[unit] = []
                units_in_order.append(unit)
            by_unit[unit].append(i)

        if len(units_in_order) == 1:
            return False, active, []

        if len(units_in_order) == 2:
            g0 = by_unit[units_in_order[0]]
            g1 = by_unit[units_in_order[1]]
            # Larger unit group goes left; ties keep first-seen unit on the left.
            left, right = (g1, g0) if len(g1) > len(g0) else (g0, g1)
            return True, sorted(left), sorted(right)

        # Three distinct units: only two axes available, so two left + one right.
        ordered = sorted(active)
        return True, ordered[:2], ordered[2:]

    def render_plots(self) -> None:
        self._clear_plot_bundles()
        self.graphics.clear()
        self.plot_bundles = []
        if self.data is None:
            return
        has_right, left_slots, right_slots = self._axis_assignment()
        names_to_draw = [self.focused_joint] if self.focused_joint else self._current_group_names()
        cells = (
            [plotting.PlotCell(self.focused_joint, 0, 0, 4, 3)]
            if self.focused_joint
            else _plot_cells_for_group(self.current_plot_group, names_to_draw)
        )
        for cell in cells:
            joint_name = cell.joint_name
            if joint_name is None or joint_name not in self.data.joint_names:
                continue
            joint_idx = self.data.joint_names.index(joint_name)
            view_box = _make_clickable_view_box(self.pg, self, joint_name)
            plot = self.graphics.addPlot(
                row=cell.row, col=cell.col, rowspan=cell.rowspan, colspan=cell.colspan,
                title=joint_name, viewBox=view_box,
            )
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setMenuEnabled(False)
            plot.setClipToView(True)
            # Show raw axis numbers; no "x0.001" / SI-prefix multipliers.
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("bottom").enableAutoSIPrefix(False)

            right_view = None
            resize_cb = None
            if has_right:
                right_view, resize_cb = self._make_right_view(plot)
            else:
                plot.hideAxis("right")

            left_curves = []
            right_curves = []
            left_sample_items = []
            right_sample_items = []
            scatter_item = None
            if self.view_mode == "scatter":
                scatter_item = self.pg.ScatterPlotItem(size=4)
                plot.addItem(scatter_item)
            else:
                for slot_idx in left_slots:
                    curve = plot.plot(pen=self._slot_pen(slot_idx, "None"))
                    curve.setClipToView(True)
                    left_curves.append((slot_idx, curve))
                    sample_item = self.pg.PlotDataItem(pen=None, symbol="o", symbolSize=4)
                    plot.addItem(sample_item)
                    left_sample_items.append((slot_idx, sample_item))
                for slot_idx in right_slots:
                    curve = self.pg.PlotCurveItem(pen=self._slot_pen(slot_idx, "None"))
                    if right_view is not None:
                        right_view.addItem(curve)
                    right_curves.append((slot_idx, curve))
                    sample_item = self.pg.PlotDataItem(pen=None, symbol="o", symbolSize=4)
                    if right_view is not None:
                        right_view.addItem(sample_item)
                    right_sample_items.append((slot_idx, sample_item))

            self.plot_bundles.append(
                _PlotBundle(
                    plot=plot,
                    joint_idx=joint_idx,
                    left_curves=left_curves,
                    right_curves=right_curves,
                    left_sample_items=left_sample_items,
                    right_sample_items=right_sample_items,
                    right_view=right_view,
                    resize_callback=resize_cb,
                    scatter_item=scatter_item,
                )
            )
        self.update_curves()

    def _curve_xy(self, slot_idx: int, joint_idx: int):
        state = self.slot_states[slot_idx]
        source = self.signal_fields.get(state.signal)
        if source is None:
            return None, None
        raw = np.asarray(source[:, joint_idx], dtype=float)
        filtered = sp.apply_filter(raw, fs=self.fs, cfg=state.filter_cfg)
        if self.view_mode == "time":
            return self.data.time, filtered
        time_range = self._current_time_range()
        if time_range is not None:
            _, filtered = _apply_time_range(self.data.time, filtered, time_range)
        return sp.compute_spectrum(filtered, fs=self.fs, cfg=self.fft_cfg)

    def _update_sample_item(self, item, slot_idx: int, x, y) -> None:
        sample_hz = self._current_sample_hz()
        if sample_hz is None or x is None or y is None:
            item.setData([], [])
            return
        sample_x, sample_y = _sample_xy_for_hz(np.asarray(x, dtype=float), np.asarray(y, dtype=float), sample_hz)
        color = _slot_color_rgb(self.slot_states[slot_idx])
        overlay_color = _sample_overlay_color(color, self._current_sample_color())
        item.setData(
            sample_x,
            sample_y,
            pen=self.pg.mkPen(overlay_color, width=1.2),
            symbol="o",
            symbolSize=4,
            symbolPen=self.pg.mkPen(overlay_color),
            symbolBrush=self.pg.mkBrush(overlay_color[0], overlay_color[1], overlay_color[2], 180),
        )

    @staticmethod
    def _y_range(arrays: list[np.ndarray]) -> tuple[float, float] | None:
        finite = [a[np.isfinite(a)] for a in arrays if a is not None and len(a)]
        finite = [a for a in finite if a.size]
        if not finite:
            return None
        allv = np.concatenate(finite)
        lo, hi = float(np.min(allv)), float(np.max(allv))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        margin = (hi - lo) * 0.05
        return lo - margin, hi + margin

    def update_curves(self) -> None:
        if self.data is None:
            return
        if self.view_mode == "scatter":
            self._update_scatter_items()
            return
        _, left_slots, right_slots = self._axis_assignment()
        left_label = self._axis_label_for_slots(left_slots)
        right_label = self._axis_label_for_slots(right_slots)
        time_range = self._current_time_range()
        for bundle in self.plot_bundles:
            left_arrays = []
            right_arrays = []
            left_xy_by_slot = {}
            right_xy_by_slot = {}
            for slot_idx, curve in bundle.left_curves:
                x, y = self._curve_xy(slot_idx, bundle.joint_idx)
                if self.view_mode == "time" and x is not None and y is not None:
                    x, y = _apply_time_range(x, y, time_range)
                curve.setPen(self._slot_pen(slot_idx, self.slot_states[slot_idx].signal))
                curve.setData([] if x is None else x, [] if y is None else y)
                left_xy_by_slot[slot_idx] = (x, y)
                if y is not None:
                    left_arrays.append(y)
            for slot_idx, curve in bundle.right_curves:
                x, y = self._curve_xy(slot_idx, bundle.joint_idx)
                if self.view_mode == "time" and x is not None and y is not None:
                    x, y = _apply_time_range(x, y, time_range)
                curve.setPen(self._slot_pen(slot_idx, self.slot_states[slot_idx].signal))
                curve.setData([] if x is None else x, [] if y is None else y)
                right_xy_by_slot[slot_idx] = (x, y)
                if y is not None:
                    right_arrays.append(y)
            for slot_idx, sample_item in bundle.left_sample_items:
                x, y = left_xy_by_slot.get(slot_idx, (None, None))
                self._update_sample_item(sample_item, slot_idx, x, y)
            for slot_idx, sample_item in bundle.right_sample_items:
                x, y = right_xy_by_slot.get(slot_idx, (None, None))
                self._update_sample_item(sample_item, slot_idx, x, y)

            if self.view_mode == "time":
                bundle.plot.setLabel("bottom", "time [s]")
                if time_range is None:
                    bundle.plot.enableAutoRange(axis="x", enable=True)
                else:
                    bundle.plot.setXRange(*time_range, padding=0.0)
            else:
                bundle.plot.setLabel("bottom", "frequency [Hz]")
                bundle.plot.setXRange(
                    float(self.fmin_spin.value()),
                    float(self.fmax_spin.value()),
                    padding=0.0,
                )
            bundle.plot.setLabel("left", left_label)

            left_limits = self._limit_values_for_signals(
                [self.slot_states[idx].signal for idx in left_slots],
                bundle.joint_idx,
            )
            left_range = _range_with_limit_values(left_arrays, left_limits)
            if bundle.right_view is not None:
                bundle.plot.setLabel("right", right_label)
                right_limits = self._limit_values_for_signals(
                    [self.slot_states[idx].signal for idx in right_slots],
                    bundle.joint_idx,
                )
                right_range = _range_with_limit_values(right_arrays, right_limits)
                if self.view_mode == "time" and left_range is not None and right_range is not None:
                    left_range, right_range = _align_zero_y_ranges(left_range, right_range)
                if left_range is not None:
                    bundle.plot.setYRange(*left_range, padding=0.0)
                if right_range is not None:
                    bundle.right_view.setYRange(*right_range, padding=0.0)
            elif left_range is not None:
                bundle.plot.setYRange(*left_range, padding=0.0)
            self._refresh_limit_items(bundle, left_slots, right_slots)

    def _update_scatter_items(self) -> None:
        x_signal = self.slot_states[0].signal
        y_signal = self.slot_states[1].signal
        x_label = SIGNAL_LABEL_BY_FIELD.get(x_signal, x_signal)
        y_label = SIGNAL_LABEL_BY_FIELD.get(y_signal, y_signal)
        y_color = _slot_color_rgb(self.slot_states[1])
        brush = self.pg.mkBrush(y_color[0], y_color[1], y_color[2], 120)
        for bundle in self.plot_bundles:
            if bundle.scatter_item is None:
                continue
            x, y = _scatter_xy_for_slots(
                self.signal_fields,
                self.slot_states,
                bundle.joint_idx,
                self.fs,
                times=self.data.time,
                time_range=self._current_time_range(),
            )
            bundle.scatter_item.setData(x=x, y=y, pen=None, brush=brush)
            bundle.plot.setLabel("bottom", x_label)
            bundle.plot.setLabel("left", y_label)
            x_limits = self._limit_values_for_signals([x_signal], bundle.joint_idx, axis="x")
            y_limits = self._limit_values_for_signals([y_signal], bundle.joint_idx, axis="y")
            zero_center = self.scatter_zero_center_check.isChecked()
            x_range = _range_with_limit_values([x], x_limits, zero_center=zero_center)
            y_range = _range_with_limit_values([y], y_limits, zero_center=zero_center)
            if x_range is not None:
                bundle.plot.setXRange(*x_range, padding=0.0)
            if y_range is not None:
                bundle.plot.setYRange(*y_range, padding=0.0)
            self._refresh_limit_items(bundle, [0, 1], [])

    def _on_signal_changed(self, slot_idx: int, text: str) -> None:
        self.slot_states[slot_idx].signal = text
        self._update_legend()
        # Signal change can alter unit grouping or scatter labels, so rebuild.
        if self.data is not None:
            self.render_plots()

    def _on_line_style_changed(self, slot_idx: int, style: str) -> None:
        self.slot_states[slot_idx].line_style = style
        self._update_legend()
        self.update_curves()

    def _on_color_changed(self, slot_idx: int, color: tuple[int, int, int] | None) -> None:
        self.slot_states[slot_idx].color_override = color
        self._update_legend()
        self.update_curves()

    def _on_plot_group_changed(self, text: str) -> None:
        self.current_plot_group = text
        self.focused_joint = None
        if self.data is not None:
            self.render_plots()

    def _rebuild_param_container(self, slot_idx: int) -> None:
        QtW = self.QtWidgets
        container = self.slot_param_containers[slot_idx]
        layout = container.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        cfg = self.slot_states[slot_idx].filter_cfg
        kind = cfg.kind

        if kind == "ma":
            window_spin = QtW.QSpinBox(); window_spin.setRange(1, 500); window_spin.setValue(cfg.window)
            layout.addWidget(QtW.QLabel("window [samples]")); layout.addWidget(window_spin)
            window_spin.valueChanged.connect(
                lambda v, idx=slot_idx: self._on_filter_param_changed(idx, window=v)
            )
        elif kind in ("lpf", "hpf"):
            fc_spin = QtW.QDoubleSpinBox()
            fc_spin.setRange(0.01, 1e6)
            fc_spin.setDecimals(2)
            fc_spin.setValue(cfg.fc)
            order_spin = QtW.QSpinBox(); order_spin.setRange(2, 10); order_spin.setValue(cfg.order)
            phase_combo = QtW.QComboBox()
            phase_combo.addItems([label for label, _ in PHASE_CHOICES])
            phase_combo.setCurrentText(PHASE_VALUE_TO_LABEL.get(cfg.phase, PHASE_CHOICES[0][0]))
            layout.addWidget(QtW.QLabel("cutoff [Hz]")); layout.addWidget(fc_spin)
            layout.addWidget(QtW.QLabel("order")); layout.addWidget(order_spin)
            layout.addWidget(QtW.QLabel("phase")); layout.addWidget(phase_combo)
            fc_spin.valueChanged.connect(lambda v, idx=slot_idx: self._on_filter_param_changed(idx, fc=v))
            order_spin.valueChanged.connect(lambda v, idx=slot_idx: self._on_filter_param_changed(idx, order=v))
            phase_combo.currentTextChanged.connect(
                lambda label, idx=slot_idx: self._on_filter_param_changed(idx, phase=PHASE_LABEL_TO_VALUE[label])
            )
        layout.addStretch(1)

    def _on_filter_kind_changed(self, slot_idx: int, kind: str) -> None:
        cfg = self.slot_states[slot_idx].filter_cfg
        self.slot_states[slot_idx].filter_cfg = sp.FilterCfg(
            kind=kind, window=cfg.window, fc=cfg.fc, order=cfg.order, phase=cfg.phase,
        )
        self._rebuild_param_container(slot_idx)
        self._update_legend()
        self.update_curves()

    def _on_filter_param_changed(
        self, slot_idx: int,
        window: int | None = None, fc: float | None = None,
        order: int | None = None, phase: str | None = None,
    ) -> None:
        cfg = self.slot_states[slot_idx].filter_cfg
        self.slot_states[slot_idx].filter_cfg = sp.FilterCfg(
            kind=cfg.kind,
            window=window if window is not None else cfg.window,
            fc=fc if fc is not None else cfg.fc,
            order=order if order is not None else cfg.order,
            phase=phase if phase is not None else cfg.phase,
        )
        self.update_curves()

    def _on_view_mode_changed(self, _checked: bool = False) -> None:
        if self.view_scatter_radio.isChecked():
            new_mode = "scatter"
        elif self.view_freq_radio.isChecked():
            new_mode = "frequency"
        else:
            new_mode = "time"
        if new_mode == self.view_mode:
            return
        self.view_mode = new_mode
        self._sync_controls_for_view_mode()
        # Time uses dual-axis (per-unit); frequency/scatter collapse to one
        # axis structure, so the plots must be rebuilt.
        if self.data is not None:
            self.render_plots()

    def _on_freq_range_changed(self, _value: float = 0.0) -> None:
        if self.fmin_spin.value() >= self.fmax_spin.value():
            return
        if self.view_mode != "frequency":
            return
        # X-range change only; spectra are unchanged, so just rescale the axis
        # instead of recomputing every FFT.
        fmin = float(self.fmin_spin.value())
        fmax = float(self.fmax_spin.value())
        for bundle in self.plot_bundles:
            bundle.plot.setXRange(fmin, fmax, padding=0.0)

    def _on_time_range_changed(self, _value: float = 0.0) -> None:
        time_range = self._current_time_range()
        if time_range is None:
            return
        self.update_curves()

    def _on_scatter_opt_changed(self, *_args) -> None:
        if self.view_mode == "scatter":
            self.update_curves()

    def _on_sample_marker_changed(self, *_args) -> None:
        self._update_legend()
        if self.view_mode == "time":
            self.update_curves()

    def _on_limit_opt_changed(self, *_args) -> None:
        self.update_curves()

    def _on_limit_robot_changed(self, *_args) -> None:
        self.robot_limits_available = _robot_limits_available(self.limit_robot_combo.currentText().lower())
        self._sync_controls_for_view_mode()
        self.reload_data()

    def _on_save_size_changed(self, label: str) -> None:
        is_custom = PNG_SIZE_BY_LABEL.get(label) is None
        self.save_width_spin.setVisible(is_custom)
        self.save_width_spin.setEnabled(is_custom)
        self.save_size_x_label.setVisible(is_custom)
        self.save_height_spin.setVisible(is_custom)
        self.save_height_spin.setEnabled(is_custom)

    def _on_fft_opt_changed(self, *_args) -> None:
        self.fft_cfg = sp.FFTCfg(
            window=self.fft_window_combo.currentText(),
            scale=self.fft_scale_combo.currentText(),
            detrend=self.fft_detrend_combo.currentText(),
            psd=self.fft_psd_check.isChecked(),
            nperseg=int(self.fft_nperseg_spin.value()),
        )
        if self.view_mode == "frequency":
            self.update_curves()

    def show(self) -> None:
        self.window.show()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SBC log plot GUI with filter, FFT, scatter, and PNG export")
    p.add_argument("path", nargs="?", default="", help="Optional SBC log file path (.txt)")
    p.add_argument("--yaml", type=str, default="", help="Policy YAML config path")
    p.add_argument("--sbc-log", type=str, default="", help="SBC log file path (.txt)")
    p.add_argument("--sample-rate-hz", type=float, default=1000.0)
    p.add_argument("--time-offset", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _, QtWidgets, _ = _require_gui()
    app = QtWidgets.QApplication(sys.argv[:1] if argv is None else ["log_plot_gui", *argv])
    win = SbcLogPlotWindow(args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
