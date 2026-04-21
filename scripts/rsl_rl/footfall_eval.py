# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play-time footfall recording and plotting utilities."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import torch


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


class PlayFootfallRecorder:
    """Capture per-step foot contact states and export a footfall raster plot."""

    FIELDNAMES = [
        "session_id",
        "env_id",
        "sim_step",
        "sim_time_s",
        "FL_contact",
        "FR_contact",
        "RL_contact",
        "RR_contact",
        "FL_force_N",
        "FR_force_N",
        "RL_force_N",
        "RR_force_N",
        "base_external_force_N",
        "base_external_force_x_b",
        "base_external_force_y_b",
        "base_external_force_z_b",
        "base_external_force_dir_deg",
        "external_force_event",
    ]

    def __init__(
        self,
        output_dir: str | Path,
        env_id: int = 0,
        force_threshold: float = 5.0,
        session_id: str | None = None,
    ) -> None:
        self.root_output_dir = Path(output_dir)
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = self.root_output_dir / self.session_id
        self.env_id = int(env_id)
        self.force_threshold = float(force_threshold)
        self.rows: list[dict[str, object]] = []
        self._prev_external_force_vec: torch.Tensor | None = None
        self._external_force_active_eps = 1.0e-3
        self._external_force_resample_eps = 0.2

    @property
    def csv_path(self) -> Path:
        return self.output_dir / "footfall_samples.csv"

    @property
    def plot_path(self) -> Path:
        return self.output_dir / "footfall_pattern.png"

    def record_step(self, env, sim_step: int, sim_time_s: float) -> None:
        """Record one play step for a single environment index."""
        base_env = unwrap_env(env)

        contact_sensor = getattr(base_env, "_contact_sensor", None)
        feet_ids = getattr(base_env, "_feet_ids", None)
        if contact_sensor is None or feet_ids is None:
            return

        contact_data = getattr(contact_sensor, "data", None)
        net_forces_w = getattr(contact_data, "net_forces_w", None) if contact_data is not None else None
        if not isinstance(net_forces_w, torch.Tensor):
            return

        num_envs = int(getattr(base_env, "num_envs", net_forces_w.shape[0]))
        if self.env_id < 0 or self.env_id >= num_envs:
            return

        if isinstance(feet_ids, torch.Tensor):
            foot_indices = feet_ids.detach().to(dtype=torch.long).view(-1)
        else:
            foot_indices = torch.tensor(feet_ids, dtype=torch.long, device=net_forces_w.device).view(-1)
        if foot_indices.numel() < 4:
            return

        feet_force_vectors = net_forces_w[self.env_id, foot_indices[:4], :]
        feet_force_norm = torch.norm(feet_force_vectors, dim=-1).detach().to(device="cpu")
        base_external_force_vec = self._read_base_external_force(base_env)
        force_x = float(base_external_force_vec[0].item())
        force_y = float(base_external_force_vec[1].item())
        force_z = float(base_external_force_vec[2].item())
        planar_force_norm = math.hypot(force_x, force_y)
        force_dir_deg = math.degrees(math.atan2(force_y, force_x)) if planar_force_norm > 1.0e-8 else float("nan")
        base_external_force_norm = float(torch.norm(base_external_force_vec).item())
        external_force_event = self._is_external_force_event(base_external_force_vec)

        row = {
            "session_id": self.session_id,
            "env_id": self.env_id,
            "sim_step": int(sim_step),
            "sim_time_s": float(sim_time_s),
            "FL_contact": int(feet_force_norm[0].item() > self.force_threshold),
            "FR_contact": int(feet_force_norm[1].item() > self.force_threshold),
            "RL_contact": int(feet_force_norm[2].item() > self.force_threshold),
            "RR_contact": int(feet_force_norm[3].item() > self.force_threshold),
            "FL_force_N": float(feet_force_norm[0].item()),
            "FR_force_N": float(feet_force_norm[1].item()),
            "RL_force_N": float(feet_force_norm[2].item()),
            "RR_force_N": float(feet_force_norm[3].item()),
            "base_external_force_N": base_external_force_norm,
            "base_external_force_x_b": force_x,
            "base_external_force_y_b": force_y,
            "base_external_force_z_b": force_z,
            "base_external_force_dir_deg": force_dir_deg,
            "external_force_event": int(external_force_event),
        }
        self.rows.append(row)

    def finalize(self) -> dict[str, Path]:
        """Persist CSV and footfall plot."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv()
        self._write_plot()
        return {
            "output_dir": self.output_dir,
            "csv_path": self.csv_path,
            "plot_path": self.plot_path,
        }

    def _write_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_plot(self) -> None:
        if not self.rows:
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
        except ImportError:
            print("[INFO] Skipping footfall plot generation because matplotlib is not installed.")
            return

        rows = sorted(self.rows, key=lambda row: int(row["sim_step"]))
        times = [float(row["sim_time_s"]) for row in rows]
        if len(times) > 1:
            dt_values = [times[idx] - times[idx - 1] for idx in range(1, len(times)) if times[idx] > times[idx - 1]]
            dt = min(dt_values) if dt_values else 0.02
        else:
            dt = 0.02

        x_min = times[0]
        x_max = times[-1] + dt
        if x_max <= x_min:
            x_max = x_min + dt

        leg_order = ["FL", "FR", "RL", "RR"]
        leg_y = {leg: len(leg_order) - 1 - idx for idx, leg in enumerate(leg_order)}
        lane_height = 0.34  # thinner stance bars for clearer timing transitions

        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Normalize stance alpha from a robust force range so outliers don't flatten contrast.
        stance_force_values = [
            float(row[f"{leg}_force_N"])
            for row in rows
            for leg in leg_order
            if int(row[f"{leg}_contact"]) == 1
        ]
        if stance_force_values:
            force_min = self._percentile(stance_force_values, 0.10)
            force_max = self._percentile(stance_force_values, 0.90)
            if force_max <= force_min + 1.0e-8:
                force_min = min(stance_force_values)
                force_max = max(stance_force_values)
        else:
            force_min = self.force_threshold
            force_max = max(self.force_threshold + 1.0, force_min + 1.0)

        lane_width = x_max - x_min
        for leg in leg_order:
            y_center = leg_y[leg]
            y_bottom = y_center - lane_height / 2.0

            # Draw explicit lane border so white swing intervals are clearly visible.
            ax.broken_barh(
                [(x_min, lane_width)],
                (y_bottom, lane_height),
                facecolors="white",
                edgecolors="black",
                linewidth=0.8,
            )

            contact_key = f"{leg}_contact"
            force_key = f"{leg}_force_N"
            contacts = [bool(int(row[contact_key])) for row in rows]
            forces = [float(row[force_key]) for row in rows]
            segments_with_force = self._contact_segments_with_force(times, contacts, forces, dt)

            # Stance = black with force-coded opacity (softer step = lighter, stronger step = darker).
            for segment_start, segment_width, segment_force in segments_with_force:
                stance_alpha = self._force_to_alpha(segment_force, force_min, force_max)
                ax.broken_barh(
                    [(segment_start, segment_width)],
                    (y_bottom, lane_height),
                    facecolors=(0.0, 0.0, 0.0, stance_alpha),
                    edgecolors="white",
                    linewidth=0.6,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.8, len(leg_order) + 0.35)
        ax.set_yticks([leg_y[leg] for leg in leg_order], leg_order)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Foot")
        ax.set_title("Footfall Pattern (Descent, 18cm step height)")

        # Add dotted vertical time guides for easier temporal reading.
        duration = x_max - x_min
        if duration <= 6.0:
            tick_step = 0.5
        elif duration <= 15.0:
            tick_step = 1.0
        elif duration <= 40.0:
            tick_step = 2.0
        else:
            tick_step = 5.0
        tick_start = tick_step * int(x_min / tick_step)
        xticks = []
        tick = tick_start
        while tick <= x_max + 1.0e-9:
            xticks.append(round(tick, 3))
            tick += tick_step
        if len(xticks) >= 2:
            ax.set_xticks(xticks)
        ax.grid(axis="x", linestyle=":", color="black", linewidth=0.7, alpha=0.45)
        ax.set_axisbelow(True)

        # Mark external-force generate/resample timestamps and annotate each event.
        force_event_rows = [row for row in rows if int(row.get("external_force_event", 0)) == 1]
        y_top = len(leg_order) - 0.06
        y_mid = len(leg_order) - 0.34
        label_dx = 0.004 * max(duration, 1.0)
        for idx, event_row in enumerate(force_event_rows):
            event_time = float(event_row["sim_time_s"])
            ax.axvline(
                x=event_time,
                color="red",
                linestyle=":",
                linewidth=2.2,
                alpha=0.95,
                zorder=10,
            )

            force_mag = float(event_row.get("base_external_force_N", 0.0))
            force_dir_deg = float(event_row.get("base_external_force_dir_deg", float("nan")))
            if math.isfinite(force_dir_deg):
                label = f"{force_mag:.1f} N, {force_dir_deg:+.0f} deg_b"
            else:
                label = f"{force_mag:.1f} N, dir_b=n/a"
            label_y = y_top if idx % 2 == 0 else y_mid
            label_x = event_time + label_dx
            label_ha = "left"
            if event_time > x_max - 0.12 * duration:
                label_x = event_time - label_dx
                label_ha = "right"
            ax.text(
                label_x,
                label_y,
                label,
                color="red",
                fontsize=6.5,
                va="bottom",
                ha=label_ha,
                zorder=11,
                clip_on=False,
            )

        legend_handles = [
            Patch(facecolor="black", edgecolor="white", label="Step (darker = higher foot force)"),
            Patch(facecolor="white", edgecolor="black", label="Swing (foot in air)"),
            Line2D([0], [0], color="red", linestyle=":", linewidth=2.2, label="External force event (label: |F|, direction_b)"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            frameon=False,
        )

        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        fig.savefig(self.plot_path)

        # Show the plot after recording ends to match interactive workflow.
        try:
            plt.show()
        except Exception as err:
            print(f"[INFO] Unable to display footfall plot window: {err}")
        finally:
            plt.close(fig)

    @staticmethod
    def _contact_segments(times: list[float], contacts: list[bool], dt: float) -> list[tuple[float, float]]:
        """Convert a binary contact trace into broken_barh-friendly segments."""
        if not times or not contacts or len(times) != len(contacts):
            return []

        segments: list[tuple[float, float]] = []
        start_time: float | None = None
        previous_time = times[0]

        for time_s, in_contact in zip(times, contacts):
            if in_contact and start_time is None:
                start_time = time_s
            if not in_contact and start_time is not None:
                end_time = previous_time + dt
                segments.append((start_time, max(end_time - start_time, 1.0e-6)))
                start_time = None
            previous_time = time_s

        if start_time is not None:
            end_time = previous_time + dt
            segments.append((start_time, max(end_time - start_time, 1.0e-6)))

        return segments

    @staticmethod
    def _contact_segments_with_force(
        times: list[float],
        contacts: list[bool],
        forces: list[float],
        dt: float,
    ) -> list[tuple[float, float, float]]:
        """Convert contact trace into stance segments with mean stance force."""
        if not times or not contacts or not forces:
            return []
        if len(times) != len(contacts) or len(times) != len(forces):
            return []

        segments: list[tuple[float, float, float]] = []
        start_idx: int | None = None

        for idx, in_contact in enumerate(contacts):
            if in_contact and start_idx is None:
                start_idx = idx
            if not in_contact and start_idx is not None:
                end_idx = idx - 1
                start_time = times[start_idx]
                end_time = times[end_idx] + dt
                width = max(end_time - start_time, 1.0e-6)
                segment_forces = forces[start_idx:idx]
                mean_force = float(sum(segment_forces) / max(len(segment_forces), 1))
                segments.append((start_time, width, mean_force))
                start_idx = None

        if start_idx is not None:
            start_time = times[start_idx]
            end_time = times[-1] + dt
            width = max(end_time - start_time, 1.0e-6)
            segment_forces = forces[start_idx:]
            mean_force = float(sum(segment_forces) / max(len(segment_forces), 1))
            segments.append((start_time, width, mean_force))

        return segments

    @staticmethod
    def _force_to_alpha(force_value: float, force_min: float, force_max: float) -> float:
        """Map a stance force to black opacity with enhanced visual contrast."""
        alpha_min = 0.05
        alpha_max = 1.0
        if force_max <= force_min + 1.0e-8:
            return alpha_max
        ratio = (force_value - force_min) / (force_max - force_min)
        ratio = max(0.0, min(1.0, ratio))
        # Gamma > 1.0 makes low/medium forces lighter and high forces darker.
        ratio = ratio**1.8
        return alpha_min + (alpha_max - alpha_min) * ratio

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        """Compute a quantile using linear interpolation on sorted values."""
        if not values:
            return 0.0
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

    def _read_base_external_force(self, base_env) -> torch.Tensor:
        """Read the current external force applied to the base body for the selected environment."""
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
        """Detect force generation or resampling by tracking changes in base external force."""
        curr_norm = float(torch.norm(force_vec).item())
        curr_active = curr_norm > self._external_force_active_eps

        if self._prev_external_force_vec is None:
            self._prev_external_force_vec = force_vec.clone()
            return curr_active

        prev_vec = self._prev_external_force_vec
        prev_norm = float(torch.norm(prev_vec).item())
        prev_active = prev_norm > self._external_force_active_eps
        delta_norm = float(torch.norm(force_vec - prev_vec).item())

        # Event if force is newly active or an active force vector changed significantly (resampled).
        event = (curr_active and not prev_active) or (
            curr_active and prev_active and delta_norm > self._external_force_resample_eps
        )
        self._prev_external_force_vec = force_vec.clone()
        return event
