# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrapper for foot-trajectory evaluation naming."""

from __future__ import annotations

from pathlib import Path

from foot_tip_eval import PlayFootTipRecorder


class PlayFootTrajectoryRecorder(PlayFootTipRecorder):
    """Same implementation as foot-tip recorder, with trajectory-style output names."""

    @property
    def samples_csv_path(self) -> Path:  # noqa: D401
        return self.output_dir / "foot_trajectory_samples.csv"

    @property
    def summary_csv_path(self) -> Path:  # noqa: D401
        return self.output_dir / "foot_trajectory_summary.csv"

    @property
    def plot_path(self) -> Path:  # noqa: D401
        return self.output_dir / "foot_trajectory.png"

