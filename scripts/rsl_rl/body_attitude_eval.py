# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play-time body-attitude stability recording and plotting utilities."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.utils.math import euler_xyz_from_quat


def unwrap_env(env):
    """Return the deepest environment-like object reachable via common wrapper attributes."""
    current = env
    seen: set[int] = set()
    while True:
        current_id = id(current)
        if current_id in seen:
            return current
        seen.add(current_id)

        next_env = getattr(current, "unwrapped", None)
        if next_env is not None and next_env is not current:
            current = next_env
            continue

        next_env = getattr(current, "env", None)
        if next_env is not None and next_env is not current:
            current = next_env
            continue

        return current


class PlayBodyAttitudeRecorder:
    """Capture body-attitude traces and compute stability/recovery metrics."""

    SAMPLE_FIELDNAMES = [
        "session_id",
        "env_id",
        "sim_step",
        "sim_time_s",
        "roll_deg",
        "pitch_deg",
        "pitch_ref_deg",
        "pitch_error_deg",
        "tilt_deg",
        "ang_vel_x_deg_s",
        "ang_vel_y_deg_s",
        "ang_vel_xy_deg_s",
        "base_external_force_N",
        "external_force_event",
    ]
    EVENT_FIELDNAMES = [
        "session_id",
        "env_id",
        "event_idx",
        "event_time_s",
        "base_external_force_N",
        "peak_tilt_deg_in_window",
        "peak_ang_vel_xy_deg_s_in_window",
        "settle_time_s",
    ]
    SUMMARY_FIELDNAMES = [
        "session_id",
        "env_id",
        "metric",
        "value",
        "unit",
    ]

    def __init__(
        self,
        output_dir: str | Path,
        env_id: int = 0,
        roll_band_deg: float = 6.0,
        pitch_band_deg: float = 8.0,
        tilt_band_deg: float = 10.0,
        ang_vel_band_deg_s: float = 35.0,
        pitch_ref_h: float = 0.0,
        pitch_ref_d: float = 1.0,
        pitch_ref_mode: str = "ascent",
        recovery_window_s: float = 2.0,
        session_id: str | None = None,
    ) -> None:
        self.root_output_dir = Path(output_dir)
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = self.root_output_dir / self.session_id
        self.env_id = int(env_id)
        self.roll_band_deg = float(roll_band_deg)
        self.pitch_band_deg = float(pitch_band_deg)
        self.tilt_band_deg = float(tilt_band_deg)
        self.ang_vel_band_deg_s = float(ang_vel_band_deg_s)
        self.pitch_ref_h = float(pitch_ref_h)
        self.pitch_ref_d = float(pitch_ref_d)
        self.pitch_ref_mode = str(pitch_ref_mode).lower()
        self.recovery_window_s = float(recovery_window_s)
        self.pitch_ref_rad = self._compute_pitch_ref_rad()
        self.pitch_ref_deg = math.degrees(self.pitch_ref_rad)

        self.rows: list[dict[str, object]] = []
        self._prev_external_force_vec: torch.Tensor | None = None
        self._external_force_active_eps = 1.0e-3
        self._external_force_resample_eps = 0.2

    @property
    def samples_csv_path(self) -> Path:
        return self.output_dir / "body_attitude_samples.csv"

    @property
    def events_csv_path(self) -> Path:
        return self.output_dir / "body_attitude_events.csv"

    @property
    def summary_csv_path(self) -> Path:
        return self.output_dir / "body_attitude_summary.csv"

    @property
    def plot_path(self) -> Path:
        return self.output_dir / "body_attitude_stability.png"

    def record_step(self, env, sim_step: int, sim_time_s: float) -> None:
        """Record one play step for one environment index."""
        base_env = unwrap_env(env)
        robot = getattr(base_env, "_robot", None)
        if robot is None:
            return

        robot_data = getattr(robot, "data", None)
        if robot_data is None:
            return

        root_quat_w = getattr(robot_data, "root_quat_w", None)
        root_ang_vel_b = getattr(robot_data, "root_ang_vel_b", None)
        if not isinstance(root_quat_w, torch.Tensor) or not isinstance(root_ang_vel_b, torch.Tensor):
            return
        if root_quat_w.ndim != 2 or root_quat_w.shape[1] != 4:
            return
        if root_ang_vel_b.ndim != 2 or root_ang_vel_b.shape[1] < 2:
            return
        if self.env_id < 0 or self.env_id >= root_quat_w.shape[0]:
            return

        quat = root_quat_w[self.env_id : self.env_id + 1, :]
        roll_rad_t, pitch_rad_t, _ = euler_xyz_from_quat(quat)
        roll_rad = float(roll_rad_t[0].item())
        pitch_rad = float(pitch_rad_t[0].item())
        pitch_error_rad = pitch_rad - self.pitch_ref_rad
        tilt_rad = math.sqrt(roll_rad * roll_rad + pitch_error_rad * pitch_error_rad)

        ang_vel_vec = root_ang_vel_b[self.env_id, :].detach().to(device="cpu")
        ang_vel_x_deg_s = math.degrees(float(ang_vel_vec[0].item()))
        ang_vel_y_deg_s = math.degrees(float(ang_vel_vec[1].item()))
        ang_vel_xy_deg_s = math.sqrt(ang_vel_x_deg_s * ang_vel_x_deg_s + ang_vel_y_deg_s * ang_vel_y_deg_s)

        base_external_force_vec = self._read_base_external_force(base_env)
        base_external_force_norm = float(torch.norm(base_external_force_vec).item())
        external_force_event = self._is_external_force_event(base_external_force_vec)

        self.rows.append(
            {
                "session_id": self.session_id,
                "env_id": self.env_id,
                "sim_step": int(sim_step),
                "sim_time_s": float(sim_time_s),
                "roll_deg": math.degrees(roll_rad),
                "pitch_deg": math.degrees(pitch_rad),
                "pitch_ref_deg": self.pitch_ref_deg,
                "pitch_error_deg": math.degrees(pitch_error_rad),
                "tilt_deg": math.degrees(tilt_rad),
                "ang_vel_x_deg_s": ang_vel_x_deg_s,
                "ang_vel_y_deg_s": ang_vel_y_deg_s,
                "ang_vel_xy_deg_s": ang_vel_xy_deg_s,
                "base_external_force_N": base_external_force_norm,
                "external_force_event": int(external_force_event),
            }
        )

    def finalize(self) -> dict[str, Path]:
        """Write CSV and plot outputs for body-attitude analysis."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = sorted(self.rows, key=lambda row: int(row["sim_step"]))
        rows = self._apply_ascent_pitch_ref_sign(rows)
        self._write_samples_csv(rows)
        event_rows = self._compute_event_metrics(rows)
        self._write_events_csv(event_rows)
        summary_rows = self._compute_summary(rows, event_rows)
        self._write_summary_csv(summary_rows)
        self._write_plot(rows, event_rows, summary_rows)
        return {
            "output_dir": self.output_dir,
            "samples_csv_path": self.samples_csv_path,
            "events_csv_path": self.events_csv_path,
            "summary_csv_path": self.summary_csv_path,
            "plot_path": self.plot_path,
        }

    def _apply_ascent_pitch_ref_sign(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Auto-resolve pitch-reference sign for ascent mode from observed pitch trace."""
        if not rows or self.pitch_ref_mode != "ascent":
            return rows

        pitch_values = [float(row["pitch_deg"]) for row in rows if self._is_finite_number(row.get("pitch_deg"))]
        if not pitch_values:
            return rows

        # Ignore near-zero samples when available to reduce flat-ground startup bias.
        significant = [value for value in pitch_values if abs(value) >= 5.0]
        source = significant if significant else pitch_values
        median_pitch = self._percentile(source, 0.5)
        if not math.isfinite(median_pitch):
            return rows

        desired_sign = -1.0 if median_pitch < 0.0 else 1.0
        resolved_pitch_ref_deg = desired_sign * abs(self.pitch_ref_deg)
        if abs(resolved_pitch_ref_deg - self.pitch_ref_deg) <= 1.0e-6:
            return rows

        self.pitch_ref_deg = resolved_pitch_ref_deg
        self.pitch_ref_rad = math.radians(resolved_pitch_ref_deg)

        # Recompute derived quantities with the resolved reference sign.
        for row in rows:
            pitch_deg = float(row["pitch_deg"])
            roll_deg = float(row["roll_deg"])
            pitch_error_deg = pitch_deg - self.pitch_ref_deg
            row["pitch_ref_deg"] = self.pitch_ref_deg
            row["pitch_error_deg"] = pitch_error_deg
            row["tilt_deg"] = math.sqrt(roll_deg * roll_deg + pitch_error_deg * pitch_error_deg)

        return rows

    def _write_samples_csv(self, rows: list[dict[str, object]]) -> None:
        with self.samples_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.SAMPLE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def _write_events_csv(self, event_rows: list[dict[str, object]]) -> None:
        with self.events_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.EVENT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(event_rows)

    def _write_summary_csv(self, summary_rows: list[dict[str, object]]) -> None:
        with self.summary_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(summary_rows)

    def _compute_event_metrics(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        if not rows:
            return []

        event_indices = [idx for idx, row in enumerate(rows) if int(row.get("external_force_event", 0)) == 1]
        event_rows: list[dict[str, object]] = []
        for event_idx, row_idx in enumerate(event_indices):
            event_time_s = float(rows[row_idx]["sim_time_s"])
            window_end_s = event_time_s + self.recovery_window_s
            window_rows = [
                row
                for row in rows
                if event_time_s <= float(row["sim_time_s"]) <= window_end_s
            ]
            if not window_rows:
                continue

            peak_tilt_deg = max(float(row["tilt_deg"]) for row in window_rows)
            peak_ang_vel_xy_deg_s = max(float(row["ang_vel_xy_deg_s"]) for row in window_rows)
            settle_time_s = self._compute_settle_time(rows, start_row_idx=row_idx, event_time_s=event_time_s)

            event_rows.append(
                {
                    "session_id": self.session_id,
                    "env_id": self.env_id,
                    "event_idx": event_idx,
                    "event_time_s": event_time_s,
                    "base_external_force_N": float(rows[row_idx]["base_external_force_N"]),
                    "peak_tilt_deg_in_window": peak_tilt_deg,
                    "peak_ang_vel_xy_deg_s_in_window": peak_ang_vel_xy_deg_s,
                    "settle_time_s": settle_time_s,
                }
            )

        return event_rows

    def _compute_settle_time(self, rows: list[dict[str, object]], start_row_idx: int, event_time_s: float) -> float:
        window_end_s = event_time_s + self.recovery_window_s
        for row in rows[start_row_idx:]:
            t = float(row["sim_time_s"])
            if t > window_end_s:
                break
            if self._is_stable_row(row):
                return t - event_time_s
        return float("nan")

    def _is_stable_row(self, row: dict[str, object]) -> bool:
        roll_abs = abs(float(row["roll_deg"]))
        pitch_error_abs = abs(float(row["pitch_error_deg"]))
        tilt = abs(float(row["tilt_deg"]))
        ang_vel_xy = abs(float(row["ang_vel_xy_deg_s"]))
        return (
            roll_abs <= self.roll_band_deg
            and pitch_error_abs <= self.pitch_band_deg
            and tilt <= self.tilt_band_deg
            and ang_vel_xy <= self.ang_vel_band_deg_s
        )

    def _compute_summary(
        self, rows: list[dict[str, object]], event_rows: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        summary_rows: list[dict[str, object]] = []
        if not rows:
            return summary_rows

        def add_metric(metric: str, value: float, unit: str):
            summary_rows.append(
                {
                    "session_id": self.session_id,
                    "env_id": self.env_id,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                }
            )

        roll_abs = [abs(float(row["roll_deg"])) for row in rows]
        pitch_err_abs = [abs(float(row["pitch_error_deg"])) for row in rows]
        tilt = [abs(float(row["tilt_deg"])) for row in rows]
        ang_vel_xy = [abs(float(row["ang_vel_xy_deg_s"])) for row in rows]

        add_metric("pitch_ref_deg", self.pitch_ref_deg, "deg")
        add_metric("roll_abs_rms_deg", self._rms(roll_abs), "deg")
        add_metric("pitch_error_abs_rms_deg", self._rms(pitch_err_abs), "deg")
        add_metric("tilt_rms_deg", self._rms(tilt), "deg")
        add_metric("ang_vel_xy_rms_deg_s", self._rms(ang_vel_xy), "deg/s")

        add_metric("roll_abs_p95_deg", self._percentile(roll_abs, 0.95), "deg")
        add_metric("pitch_error_abs_p95_deg", self._percentile(pitch_err_abs, 0.95), "deg")
        add_metric("tilt_p95_deg", self._percentile(tilt, 0.95), "deg")
        add_metric("ang_vel_xy_p95_deg_s", self._percentile(ang_vel_xy, 0.95), "deg/s")

        add_metric("roll_out_of_band_pct", self._out_of_band_pct(roll_abs, self.roll_band_deg), "pct")
        add_metric("pitch_error_out_of_band_pct", self._out_of_band_pct(pitch_err_abs, self.pitch_band_deg), "pct")
        add_metric("tilt_out_of_band_pct", self._out_of_band_pct(tilt, self.tilt_band_deg), "pct")
        add_metric("ang_vel_xy_out_of_band_pct", self._out_of_band_pct(ang_vel_xy, self.ang_vel_band_deg_s), "pct")

        settle_times = [float(row["settle_time_s"]) for row in event_rows if self._is_finite_number(row["settle_time_s"])]
        add_metric("external_force_event_count", float(len(event_rows)), "count")
        add_metric("mean_settle_time_s", self._mean(settle_times), "s")
        add_metric("p95_settle_time_s", self._percentile(settle_times, 0.95), "s")

        return summary_rows

    def _write_plot(
        self, rows: list[dict[str, object]], event_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]
    ) -> None:
        if not rows:
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MultipleLocator
        except ImportError:
            print("[INFO] Skipping body-attitude plot generation because matplotlib is not installed.")
            return

        times = [float(row["sim_time_s"]) for row in rows]
        roll = [float(row["roll_deg"]) for row in rows]
        pitch = [float(row["pitch_deg"]) for row in rows]
        pitch_ref = [float(row["pitch_ref_deg"]) for row in rows]
        pitch_error = [float(row["pitch_error_deg"]) for row in rows]
        tilt = [float(row["tilt_deg"]) for row in rows]
        ang_vel_x = [float(row["ang_vel_x_deg_s"]) for row in rows]
        ang_vel_y = [float(row["ang_vel_y_deg_s"]) for row in rows]
        ang_vel_xy = [float(row["ang_vel_xy_deg_s"]) for row in rows]
        force_event_times = [float(row["sim_time_s"]) for row in rows if int(row.get("external_force_event", 0)) == 1]
        x_min = times[0]
        x_max = times[-1]
        if x_max <= x_min:
            x_max = x_min + 1.0e-6

        summary_map = {str(row["metric"]): float(row["value"]) for row in summary_rows if self._is_finite_number(row.get("value"))}

        fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), dpi=160, sharex=False)
        title_y = 0.935
        fig.suptitle("Roll/Pitch Stability", fontsize=14, fontweight="bold", y=title_y)

        ax1 = axes[0]
        ax1.plot(times, roll, color="#1f4aff", linewidth=1.2, label="Roll")
        ax1.plot(times, pitch, color="#ff6a00", linewidth=1.2, label="Pitch")
        ax1.plot(times, pitch_ref, color="black", linestyle="--", linewidth=1.0, label="Pitch ref")
        ax1.axhline(+self.roll_band_deg, color="#1f4aff", linestyle=":", linewidth=0.9, alpha=0.8)
        ax1.axhline(-self.roll_band_deg, color="#1f4aff", linestyle=":", linewidth=0.9, alpha=0.8)
        ax1.axhline(self.pitch_ref_deg + self.pitch_band_deg, color="#ff6a00", linestyle=":", linewidth=0.9, alpha=0.8)
        ax1.axhline(self.pitch_ref_deg - self.pitch_band_deg, color="#ff6a00", linestyle=":", linewidth=0.9, alpha=0.8)
        self._draw_event_lines(ax1, force_event_times)
        self._annotate_event_labels(ax1, event_rows, x_min=times[0], x_max=times[-1])
        self._annotate_hline_value(ax1, +self.roll_band_deg, f"+{self.roll_band_deg:.1f} deg", "#1f4aff", x_min, x_max)
        self._annotate_hline_value(ax1, -self.roll_band_deg, f"-{self.roll_band_deg:.1f} deg", "#1f4aff", x_min, x_max)
        self._annotate_hline_value(
            ax1,
            self.pitch_ref_deg + self.pitch_band_deg,
            f"{self.pitch_ref_deg + self.pitch_band_deg:.1f} deg",
            "#ff6a00",
            x_min,
            x_max,
        )
        self._annotate_hline_value(
            ax1,
            self.pitch_ref_deg - self.pitch_band_deg,
            f"{self.pitch_ref_deg - self.pitch_band_deg:.1f} deg",
            "#ff6a00",
            x_min,
            x_max,
        )
        ax1.set_ylabel("Angle (deg)")
        ax1.set_title("1) Roll/Pitch tracking")
        ax1.set_xlim(x_min, x_max)
        ax1.xaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_major_locator(MultipleLocator(10.0))
        ax1.grid(axis="both", linestyle=":", linewidth=0.8, alpha=0.45)
        ax1.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            ncol=1,
            frameon=False,
            borderaxespad=0.2,
        )

        ax2 = axes[1]
        ax2.plot(times, pitch_error, color="#ff8c00", linewidth=1.2, label="Pitch error")
        ax2.plot(times, tilt, color="black", linewidth=1.2, label="Tilt magnitude")
        ax2.axhline(+self.pitch_band_deg, color="#ff8c00", linestyle=":", linewidth=0.9, alpha=0.8)
        ax2.axhline(-self.pitch_band_deg, color="#ff8c00", linestyle=":", linewidth=0.9, alpha=0.8)
        ax2.axhline(self.tilt_band_deg, color="black", linestyle="--", linewidth=1.0, alpha=0.85)
        self._draw_event_lines(ax2, force_event_times)
        self._annotate_hline_value(ax2, +self.pitch_band_deg, f"+{self.pitch_band_deg:.1f} deg", "#ff8c00", x_min, x_max)
        self._annotate_hline_value(ax2, -self.pitch_band_deg, f"-{self.pitch_band_deg:.1f} deg", "#ff8c00", x_min, x_max)
        self._annotate_hline_value(ax2, self.tilt_band_deg, f"{self.tilt_band_deg:.1f} deg", "black", x_min, x_max)
        ax2.set_ylabel("Deg")
        ax2.set_title("2) Pitch error + tilt magnitude")
        ax2.set_xlim(x_min, x_max)
        ax2.xaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_major_locator(MultipleLocator(10.0))
        ax2.grid(axis="both", linestyle=":", linewidth=0.8, alpha=0.45)
        ax2.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            ncol=1,
            frameon=False,
            borderaxespad=0.2,
        )

        ax3 = axes[2]
        ax3.plot(times, ang_vel_x, color="#31a354", linewidth=1.0, label="wx")
        ax3.plot(times, ang_vel_y, color="#9467bd", linewidth=1.0, label="wy")
        ax3.plot(times, ang_vel_xy, color="#d62728", linewidth=1.2, label="|wxy|")
        ax3.axhline(self.ang_vel_band_deg_s, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.85)
        self._draw_event_lines(ax3, force_event_times)
        self._annotate_hline_value(
            ax3,
            self.ang_vel_band_deg_s,
            f"{self.ang_vel_band_deg_s:.1f} deg/s",
            "#d62728",
            x_min,
            x_max,
        )
        ax3.set_ylabel("deg/s")
        ax3.set_title("3) Angular-rate stability")
        ax3.set_xlabel("Time (s)")
        ax3.set_xlim(x_min, x_max)
        ax3.grid(axis="both", linestyle=":", linewidth=0.8, alpha=0.45)
        ax3.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            ncol=1,
            frameon=False,
            borderaxespad=0.2,
        )

        summary_lines = [
            f"roll p95: {summary_map.get('roll_abs_p95_deg', float('nan')):.2f} deg",
            f"pitch_err p95: {summary_map.get('pitch_error_abs_p95_deg', float('nan')):.2f} deg",
            f"tilt p95: {summary_map.get('tilt_p95_deg', float('nan')):.2f} deg",
            f"|wxy| p95: {summary_map.get('ang_vel_xy_p95_deg_s', float('nan')):.2f} deg/s",
        ]
        fig.text(0.012, 0.012, "p95 (absolute, over recorded timesteps): " + " | ".join(summary_lines), fontsize=9)

        fig.tight_layout(rect=(0.0, 0.02, 1.0, title_y - 0.02))
        fig.savefig(self.plot_path)

        try:
            plt.show()
        except Exception as err:
            print(f"[INFO] Unable to display body-attitude plot window: {err}")
        finally:
            plt.close(fig)

    @staticmethod
    def _draw_event_lines(ax, event_times: list[float]) -> None:
        for t in event_times:
            ax.axvline(t, color="red", linestyle=":", linewidth=1.4, alpha=0.85)

    @staticmethod
    def _annotate_event_labels(ax, event_rows: list[dict[str, object]], x_min: float, x_max: float) -> None:
        """Add compact labels for external-force events near red dotted lines."""
        if not event_rows:
            return
        x_span = max(x_max - x_min, 1.0e-6)
        label_dx = 0.004 * x_span
        y_min, y_max = ax.get_ylim()
        y_span = max(y_max - y_min, 1.0e-6)
        y_levels = [y_max - 0.08 * y_span, y_max - 0.18 * y_span, y_max - 0.28 * y_span]

        for idx, row in enumerate(event_rows):
            event_time = float(row.get("event_time_s", float("nan")))
            force_mag = float(row.get("base_external_force_N", float("nan")))
            if not math.isfinite(event_time):
                continue

            label = f"E{idx + 1}: {force_mag:.1f}N" if math.isfinite(force_mag) else f"E{idx + 1}"
            label_x = event_time + label_dx
            label_ha = "left"
            if event_time > x_max - 0.12 * x_span:
                label_x = event_time - label_dx
                label_ha = "right"

            ax.text(
                label_x,
                y_levels[idx % len(y_levels)],
                label,
                color="red",
                fontsize=6.5,
                ha=label_ha,
                va="bottom",
                zorder=12,
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )

    @staticmethod
    def _annotate_hline_value(ax, y_value: float, label: str, color: str, x_min: float, x_max: float) -> None:
        """Add a compact label near the right end of a horizontal threshold line."""
        if not math.isfinite(y_value):
            return
        x_span = max(x_max - x_min, 1.0e-6)
        x_pos = x_max - 0.01 * x_span
        ax.text(
            x_pos,
            y_value,
            label,
            color=color,
            fontsize=6.5,
            ha="right",
            va="bottom",
            zorder=12,
            clip_on=True,
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    def _compute_pitch_ref_rad(self) -> float:
        if self.pitch_ref_mode == "flat":
            return 0.0
        run = self.pitch_ref_d if abs(self.pitch_ref_d) > 1.0e-8 else 1.0e-8
        base_angle = math.atan2(abs(self.pitch_ref_h), abs(run))
        if self.pitch_ref_mode == "ascent":
            return base_angle
        if self.pitch_ref_mode == "descent":
            return -base_angle
        raise ValueError("pitch_ref_mode must be one of: ascent, descent, flat")

    def _read_base_external_force(self, base_env) -> torch.Tensor:
        """Read current external force applied to base body for selected environment."""
        robot = getattr(base_env, "_robot", None)
        if robot is None:
            return torch.zeros(3, dtype=torch.float32)

        force_buffer = getattr(robot, "_external_force_b", None)
        if force_buffer is None:
            robot_data = getattr(robot, "data", None)
            force_buffer = getattr(robot_data, "external_force_b", None) if robot_data is not None else None
        if not isinstance(force_buffer, torch.Tensor) or force_buffer.ndim != 3:
            return torch.zeros(3, dtype=torch.float32)
        if self.env_id < 0 or self.env_id >= force_buffer.shape[0]:
            return torch.zeros(3, dtype=torch.float32)

        base_body_id = int(getattr(base_env, "_base_body_id", 0))
        if base_body_id < 0 or base_body_id >= force_buffer.shape[1]:
            base_body_id = 0
        return force_buffer[self.env_id, base_body_id, :].detach().to(device="cpu")

    def _is_external_force_event(self, force_vec: torch.Tensor) -> bool:
        curr_norm = float(torch.norm(force_vec).item())
        curr_active = curr_norm > self._external_force_active_eps
        if self._prev_external_force_vec is None:
            self._prev_external_force_vec = force_vec.clone()
            return curr_active

        prev_vec = self._prev_external_force_vec
        prev_norm = float(torch.norm(prev_vec).item())
        prev_active = prev_norm > self._external_force_active_eps
        delta_norm = float(torch.norm(force_vec - prev_vec).item())
        event = (curr_active and not prev_active) or (
            curr_active and prev_active and delta_norm > self._external_force_resample_eps
        )
        self._prev_external_force_vec = force_vec.clone()
        return event

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return float("nan")
        return sum(values) / len(values)

    @staticmethod
    def _rms(values: list[float]) -> float:
        if not values:
            return float("nan")
        return math.sqrt(sum(value * value for value in values) / len(values))

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return float("nan")
        q = max(0.0, min(1.0, float(quantile)))
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lower_idx = int(math.floor(position))
        upper_idx = int(math.ceil(position))
        if lower_idx == upper_idx:
            return ordered[lower_idx]
        upper_weight = position - lower_idx
        lower_weight = 1.0 - upper_weight
        return ordered[lower_idx] * lower_weight + ordered[upper_idx] * upper_weight

    @staticmethod
    def _out_of_band_pct(values: list[float], threshold: float) -> float:
        if not values:
            return float("nan")
        violations = sum(1 for value in values if value > threshold)
        return 100.0 * violations / len(values)

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
