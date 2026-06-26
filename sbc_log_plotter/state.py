# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""Shared state types for log plotting.

This module intentionally has no Qt, matplotlib, robot-runtime config, or non-SBC log
runtime imports so save/SBC plotting can use it in a small Python distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import signal_pipeline as sp


ViewMode = Literal["time", "scatter", "frequency"]
InputMode = Literal["sbc"]


@dataclass
class SlotState:
    signal: str = "None"
    line_style: str = "-"
    color_override: tuple[int, int, int] | None = None
    filter_cfg: sp.FilterCfg = field(default_factory=sp.FilterCfg)
