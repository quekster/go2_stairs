# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play-time duty-factor and phase-offset recording utilities."""

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


class PlayDutyPhaseRecorder:
    """Capture per-step contacts and compute duty factor + phase offsets."""

    LEG_NAMES = ("FL", "FR", "RL", "RR")
    SAMPLE_FIELDNAMES = [
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
    ]
    CYCLE_FIELDNAMES = [
        "session_id",
        "env_id",
        "cycle_idx",
        "start_step",
        "end_step",
        "start_time_s",
        "end_time_s",
        "cycle_period_s",
        "FL_duty_factor",
        "FR_duty_factor",
        "RL_duty_factor",
        "RR_duty_factor",
        "FL_phase_offset",
        "FR_phase_offset",
        "RL_phase_offset",
        "RR_phase_offset",
    ]
    SUMMARY_FIELDNAMES = ["session_id", "env_id", "metric", "mean", "std", "count"]

    def __init__(
        self,
        output_dir: str | Path,
        env_id: int = 0,
        force_threshold: float = 5.0,
        phase_reference_leg: str = "FL",
        session_id: str | None = None,
    ) -> None:
        self.root_output_dir = Path(output_dir)
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = self.root_output_dir / self.session_id
        self.env_id = int(env_id)
        self.force_threshold = float(force_threshold)
        reference_leg = phase_reference_leg.upper()
        if reference_leg not in self.LEG_NAMES:
            raise ValueError(
                f"Invalid phase_reference_leg='{phase_reference_leg}'. "
                f"Expected one of {self.LEG_NAMES}."
            )
        self.phase_reference_leg = reference_leg
        self.rows: list[dict[str, object]] = []

    @property
    def samples_csv_path(self) -> Path:
        return self.output_dir / "duty_phase_samples.csv"

    @property
    def cycles_csv_path(self) -> Path:
        return self.output_dir / "duty_phase_cycles.csv"

    @property
    def summary_csv_path(self) -> Path:
        return self.output_dir / "duty_phase_summary.csv"

    @property
    def plot_path(self) -> Path:
        return self.output_dir / "duty_phase_plot.png"

    def record_step(self, env, sim_step: int, sim_time_s: float) -> None:
        """Record one play step for one environment index."""
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

        force_vec = net_forces_w[self.env_id, foot_indices[:4], :]
        force_norm = torch.norm(force_vec, dim=-1).detach().to(device="cpu")

        self.rows.append(
            {
                "session_id": self.session_id,
                "env_id": self.env_id,
                "sim_step": int(sim_step),
                "sim_time_s": float(sim_time_s),
                "FL_contact": int(force_norm[0].item() > self.force_threshold),
                "FR_contact": int(force_norm[1].item() > self.force_threshold),
                "RL_contact": int(force_norm[2].item() > self.force_threshold),
                "RR_contact": int(force_norm[3].item() > self.force_threshold),
                "FL_force_N": float(force_norm[0].item()),
                "FR_force_N": float(force_norm[1].item()),
                "RL_force_N": float(force_norm[2].item()),
                "RR_force_N": float(force_norm[3].item()),
            }
        )

    def finalize(self) -> dict[str, Path]:
        """Write CSV/plot outputs for duty-factor and phase-offset analysis."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_samples_csv()
        cycles = self._compute_cycle_metrics()
        self._write_cycles_csv(cycles)
        summary = self._compute_summary(cycles)
        self._write_summary_csv(summary)
        self._write_plot(cycles, summary)
        return {
            "output_dir": self.output_dir,
            "samples_csv_path": self.samples_csv_path,
            "cycles_csv_path": self.cycles_csv_path,
            "summary_csv_path": self.summary_csv_path,
            "plot_path": self.plot_path,
        }

    def _write_samples_csv(self) -> None:
        with self.samples_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.SAMPLE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_cycles_csv(self, cycles: list[dict[str, object]]) -> None:
        with self.cycles_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.CYCLE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(cycles)

    def _write_summary_csv(self, summary_rows: list[dict[str, object]]) -> None:
        with self.summary_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(summary_rows)

    def _compute_cycle_metrics(self) -> list[dict[str, object]]:
        if len(self.rows) < 3:
            return []

        rows = sorted(self.rows, key=lambda row: int(row["sim_step"]))
        times = [float(row["sim_time_s"]) for row in rows]
        dt = self._estimate_dt(times)
        durations = self._interval_durations(times, dt)

        contacts = {leg: [bool(int(row[f"{leg}_contact"])) for row in rows] for leg in self.LEG_NAMES}
        touchdowns = {leg: self._touchdown_indices(contacts[leg]) for leg in self.LEG_NAMES}

        ref_leg = self.phase_reference_leg
        ref_touchdowns = touchdowns[ref_leg]
        if len(ref_touchdowns) < 2:
            return []

        cycles: list[dict[str, object]] = []
        for cycle_idx in range(len(ref_touchdowns) - 1):
            start_idx = ref_touchdowns[cycle_idx]
            end_idx = ref_touchdowns[cycle_idx + 1]
            if end_idx <= start_idx:
                continue

            start_t = times[start_idx]
            end_t = times[end_idx]
            cycle_period = end_t - start_t
            if cycle_period <= 1.0e-6:
                continue

            cycle = {
                "session_id": self.session_id,
                "env_id": self.env_id,
                "cycle_idx": cycle_idx,
                "start_step": int(rows[start_idx]["sim_step"]),
                "end_step": int(rows[end_idx]["sim_step"]),
                "start_time_s": start_t,
                "end_time_s": end_t,
                "cycle_period_s": cycle_period,
            }

            for leg in self.LEG_NAMES:
                duty_factor = self._compute_duty_factor(contacts[leg], durations, start_idx, end_idx, cycle_period)
                cycle[f"{leg}_duty_factor"] = duty_factor

            for leg in self.LEG_NAMES:
                if leg == ref_leg:
                    cycle[f"{leg}_phase_offset"] = 0.0
                else:
                    touchdown_idx = self._first_touchdown_in_window(touchdowns[leg], start_idx, end_idx)
                    if touchdown_idx is None:
                        cycle[f"{leg}_phase_offset"] = float("nan")
                    else:
                        phase = (times[touchdown_idx] - start_t) / cycle_period
                        cycle[f"{leg}_phase_offset"] = max(0.0, min(phase, 1.0))

            cycles.append(cycle)

        return cycles

    def _compute_summary(self, cycles: list[dict[str, object]]) -> list[dict[str, object]]:
        metrics = [
            "FL_duty_factor",
            "FR_duty_factor",
            "RL_duty_factor",
            "RR_duty_factor",
            "FL_phase_offset",
            "FR_phase_offset",
            "RL_phase_offset",
            "RR_phase_offset",
        ]

        summary_rows: list[dict[str, object]] = []
        for metric in metrics:
            values = [float(cycle[metric]) for cycle in cycles if self._is_finite_number(cycle.get(metric))]
            mean, std = self._mean_std(values)
            summary_rows.append(
                {
                    "session_id": self.session_id,
                    "env_id": self.env_id,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "count": len(values),
                }
            )

        return summary_rows

    def _write_plot(self, cycles: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[INFO] Skipping duty/phase plot generation because matplotlib is not installed.")
            return

        _ = summary_rows  # Plot is stride-wise; summary CSV remains available separately.
        cycles_sorted = sorted(cycles, key=lambda cycle: int(cycle["cycle_idx"]))
        if not cycles_sorted:
            fig, ax = plt.subplots(figsize=(10, 3.5), dpi=160)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"No full {self.phase_reference_leg}-to-{self.phase_reference_leg} stride cycles were detected.\n"
                "Record a longer run to plot duty/phase over strides.",
                ha="center",
                va="center",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(self.plot_path)
            try:
                plt.show()
            except Exception as err:
                print(f"[INFO] Unable to display duty/phase plot window: {err}")
            finally:
                plt.close(fig)
            return

        stride_numbers = [int(cycle["cycle_idx"]) + 1 for cycle in cycles_sorted]
        display_name = {"FL": "FL", "FR": "FR", "RL": "RL", "RR": "RR"}
        colors = {"FL": "#1f4aff", "FR": "#ff6a00", "RL": "#1b8a2f", "RR": "#7a1ec2"}

        def _series(metric: str) -> list[float]:
            series = []
            for cycle in cycles_sorted:
                value = cycle.get(metric)
                if self._is_finite_number(value):
                    series.append(float(value))
                else:
                    series.append(float("nan"))
            return series

        duty_data = {leg: _series(f"{leg}_duty_factor") for leg in self.LEG_NAMES}
        phase_data = {leg: _series(f"{leg}_phase_offset") for leg in self.LEG_NAMES}

        duty_values = [value for values in duty_data.values() for value in values if math.isfinite(value)]
        if duty_values:
            duty_min = max(0.0, min(duty_values) - 0.03)
            duty_max = min(1.0, max(duty_values) + 0.03)
            if duty_max - duty_min < 0.18:
                center = 0.5 * (duty_min + duty_max)
                duty_min = max(0.0, center - 0.09)
                duty_max = min(1.0, center + 0.09)
        else:
            duty_min, duty_max = 0.3, 0.7

        fig, (ax_duty, ax_phase) = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=160)
        fig.suptitle("Duty factor + phase offsets per leg over time", fontsize=13, fontweight="bold")

        for leg in self.LEG_NAMES:
            ax_duty.plot(
                stride_numbers,
                duty_data[leg],
                marker="o",
                markersize=3.8,
                linewidth=1.2,
                color=colors[leg],
                label=display_name[leg],
            )
        ax_duty.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
        ax_duty.set_title("1) Duty factor per leg")
        ax_duty.set_xlabel("Stride number")
        ax_duty.set_ylabel("Duty factor")
        ax_duty.set_ylim(duty_min, duty_max)
        ax_duty.grid(axis="both", linestyle=":", linewidth=0.8, alpha=0.45)
        ax_duty.set_axisbelow(True)
        ax_duty.legend(loc="upper right", frameon=False)

        for leg in self.LEG_NAMES:
            label = f"{display_name[leg]} (reference)" if leg == self.phase_reference_leg else display_name[leg]
            ax_phase.plot(
                stride_numbers,
                phase_data[leg],
                marker="o",
                markersize=3.8,
                linewidth=1.2,
                color=colors[leg],
                label=label,
            )
        ax_phase.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
        ax_phase.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.8)
        ax_phase.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.8)
        ax_phase.set_title(f"2) Phase offsets relative to {self.phase_reference_leg}")
        ax_phase.set_xlabel("Stride number")
        ax_phase.set_ylabel("Phase offset (0 to 1 of stride)")
        ax_phase.set_ylim(-0.03, 1.03)
        ax_phase.grid(axis="both", linestyle=":", linewidth=0.8, alpha=0.45)
        ax_phase.set_axisbelow(True)
        ax_phase.legend(loc="upper right", frameon=False)

        if len(stride_numbers) <= 20:
            xticks = stride_numbers
        else:
            tick_step = max(1, len(stride_numbers) // 10)
            xticks = stride_numbers[::tick_step]
            if xticks[-1] != stride_numbers[-1]:
                xticks.append(stride_numbers[-1])
        ax_duty.set_xticks(xticks)
        ax_phase.set_xticks(xticks)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(self.plot_path)

        try:
            plt.show()
        except Exception as err:
            print(f"[INFO] Unable to display duty/phase plot window: {err}")
        finally:
            plt.close(fig)

    @staticmethod
    def _estimate_dt(times: list[float]) -> float:
        if len(times) < 2:
            return 0.02
        dt_values = [times[idx] - times[idx - 1] for idx in range(1, len(times)) if times[idx] > times[idx - 1]]
        return min(dt_values) if dt_values else 0.02

    @staticmethod
    def _interval_durations(times: list[float], dt_fallback: float) -> list[float]:
        if not times:
            return []
        durations = []
        for idx in range(len(times) - 1):
            durations.append(max(times[idx + 1] - times[idx], 0.0))
        durations.append(max(dt_fallback, 0.0))
        return durations

    @staticmethod
    def _touchdown_indices(contact_trace: list[bool]) -> list[int]:
        touchdowns: list[int] = []
        previous = False
        for idx, in_contact in enumerate(contact_trace):
            if in_contact and not previous:
                touchdowns.append(idx)
            previous = in_contact
        return touchdowns

    @staticmethod
    def _first_touchdown_in_window(touchdown_indices: list[int], start_idx: int, end_idx: int) -> int | None:
        for idx in touchdown_indices:
            if start_idx <= idx < end_idx:
                return idx
        return None

    @staticmethod
    def _compute_duty_factor(
        contact_trace: list[bool],
        durations: list[float],
        start_idx: int,
        end_idx: int,
        cycle_period: float,
    ) -> float:
        if cycle_period <= 1.0e-8:
            return float("nan")
        stance_time = 0.0
        for idx in range(start_idx, end_idx):
            if idx < len(contact_trace) and contact_trace[idx]:
                stance_time += durations[idx]
        return max(0.0, min(stance_time / cycle_period, 1.0))

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(max(variance, 0.0))
        return mean, std

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
