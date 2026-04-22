# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play-time foot-tip trajectory recording and plotting utilities."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate


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


class PlayFootTipRecorder:
    """Capture foot-tip trajectories and export step-length/swing-height plots."""

    LEG_NAMES = ["FL", "FR", "RL", "RR"]
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
        "FL_x_w_m",
        "FL_y_w_m",
        "FL_z_w_m",
        "FR_x_w_m",
        "FR_y_w_m",
        "FR_z_w_m",
        "RL_x_w_m",
        "RL_y_w_m",
        "RL_z_w_m",
        "RR_x_w_m",
        "RR_y_w_m",
        "RR_z_w_m",
        "FL_x_b_m",
        "FL_y_b_m",
        "FL_z_b_m",
        "FR_x_b_m",
        "FR_y_b_m",
        "FR_z_b_m",
        "RL_x_b_m",
        "RL_y_b_m",
        "RL_z_b_m",
        "RR_x_b_m",
        "RR_y_b_m",
        "RR_z_b_m",
        "FL_swing_height_m",
        "FR_swing_height_m",
        "RL_swing_height_m",
        "RR_swing_height_m",
    ]

    SUMMARY_FIELDNAMES = [
        "metric",
        "value",
        "unit",
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
        self._foot_body_ids: torch.Tensor | None = None
        self._foot_contact_ids: torch.Tensor | None = None

    @property
    def samples_csv_path(self) -> Path:
        return self.output_dir / "foot_tip_samples.csv"

    @property
    def summary_csv_path(self) -> Path:
        return self.output_dir / "foot_tip_summary.csv"

    @property
    def plot_path(self) -> Path:
        return self.output_dir / "foot_tip_trajectory.png"

    def record_step(self, env, sim_step: int, sim_time_s: float) -> None:
        """Record one play step for a single environment index."""
        base_env = unwrap_env(env)

        robot = getattr(base_env, "_robot", None)
        contact_sensor = getattr(base_env, "_contact_sensor", None)
        if robot is None:
            return

        robot_data = getattr(robot, "data", None)
        body_pos_w = getattr(robot_data, "body_pos_w", None) if robot_data is not None else None
        root_pos_w = getattr(robot_data, "root_pos_w", None) if robot_data is not None else None
        root_quat_w = getattr(robot_data, "root_quat_w", None) if robot_data is not None else None
        if not isinstance(body_pos_w, torch.Tensor) or not isinstance(root_pos_w, torch.Tensor) or not isinstance(
            root_quat_w, torch.Tensor
        ):
            return

        num_envs = int(getattr(base_env, "num_envs", body_pos_w.shape[0]))
        if self.env_id < 0 or self.env_id >= num_envs:
            return

        contact_data = getattr(contact_sensor, "data", None) if contact_sensor is not None else None
        net_forces_w = getattr(contact_data, "net_forces_w", None) if contact_data is not None else None

        foot_body_ids, foot_contact_ids = self._resolve_foot_indices(base_env, body_pos_w, net_forces_w)
        if foot_body_ids is None or foot_body_ids.numel() < 4:
            return

        feet_pos_w = body_pos_w[self.env_id, foot_body_ids[:4], :].detach()
        base_pos_w = root_pos_w[self.env_id, :].detach()
        base_quat_w = root_quat_w[self.env_id : self.env_id + 1, :].detach()
        base_quat_inv = quat_conjugate(base_quat_w).expand(feet_pos_w.shape[0], -1)
        feet_pos_b = quat_apply(base_quat_inv, feet_pos_w - base_pos_w.unsqueeze(0)).to(device="cpu")

        feet_force_norm = torch.zeros(4, dtype=torch.float32)
        if isinstance(net_forces_w, torch.Tensor) and foot_contact_ids is not None and foot_contact_ids.numel() >= 4:
            if self.env_id < net_forces_w.shape[0] and int(torch.max(foot_contact_ids).item()) < net_forces_w.shape[1]:
                feet_force_vectors = net_forces_w[self.env_id, foot_contact_ids[:4], :]
                feet_force_norm = torch.norm(feet_force_vectors, dim=-1).detach().to(device="cpu")

        row = {
            "session_id": self.session_id,
            "env_id": self.env_id,
            "sim_step": int(sim_step),
            "sim_time_s": float(sim_time_s),
        }
        for leg_idx, leg in enumerate(self.LEG_NAMES):
            row[f"{leg}_contact"] = int(float(feet_force_norm[leg_idx].item()) > self.force_threshold)
            row[f"{leg}_force_N"] = float(feet_force_norm[leg_idx].item())
            row[f"{leg}_x_w_m"] = float(feet_pos_w[leg_idx, 0].item())
            row[f"{leg}_y_w_m"] = float(feet_pos_w[leg_idx, 1].item())
            row[f"{leg}_z_w_m"] = float(feet_pos_w[leg_idx, 2].item())
            row[f"{leg}_x_b_m"] = float(feet_pos_b[leg_idx, 0].item())
            row[f"{leg}_y_b_m"] = float(feet_pos_b[leg_idx, 1].item())
            row[f"{leg}_z_b_m"] = float(feet_pos_b[leg_idx, 2].item())
        self.rows.append(row)

    def finalize(self) -> dict[str, Path]:
        """Persist CSV and foot-tip trajectory plot."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = sorted(self.rows, key=lambda row: int(row["sim_step"]))
        rows, z_ref_by_leg = self._attach_swing_heights(rows)
        self._write_samples_csv(rows)
        summary_rows = self._compute_summary(rows, z_ref_by_leg)
        self._write_summary_csv(summary_rows)
        self._write_plot(rows, z_ref_by_leg)
        return {
            "output_dir": self.output_dir,
            "samples_csv_path": self.samples_csv_path,
            "summary_csv_path": self.summary_csv_path,
            "plot_path": self.plot_path,
        }

    def _attach_swing_heights(self, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, float]]:
        if not rows:
            return rows, {leg: 0.0 for leg in self.LEG_NAMES}

        z_ref_by_leg: dict[str, float] = {}
        for leg in self.LEG_NAMES:
            stance_z = [
                float(row[f"{leg}_z_b_m"])
                for row in rows
                if int(row.get(f"{leg}_contact", 0)) == 1 and self._is_finite_number(row.get(f"{leg}_z_b_m"))
            ]
            all_z = [float(row[f"{leg}_z_b_m"]) for row in rows if self._is_finite_number(row.get(f"{leg}_z_b_m"))]
            source = stance_z if stance_z else all_z
            z_ref_by_leg[leg] = self._percentile(source, 0.5) if source else 0.0

        with_swing: list[dict[str, object]] = []
        for row in rows:
            row_out = dict(row)
            for leg in self.LEG_NAMES:
                z_now = float(row_out[f"{leg}_z_b_m"])
                row_out[f"{leg}_swing_height_m"] = z_now - z_ref_by_leg[leg]
            with_swing.append(row_out)
        return with_swing, z_ref_by_leg

    def _compute_summary(
        self, rows: list[dict[str, object]], z_ref_by_leg: dict[str, float]
    ) -> list[dict[str, object]]:
        summary_rows: list[dict[str, object]] = []

        def add_metric(metric: str, value: float, unit: str) -> None:
            summary_rows.append({"metric": metric, "value": value, "unit": unit})

        add_metric("sample_count", float(len(rows)), "count")
        for leg in self.LEG_NAMES:
            add_metric(f"{leg}_stance_z_ref_b", z_ref_by_leg.get(leg, 0.0), "m")
            swing_values = [
                float(row[f"{leg}_swing_height_m"])
                for row in rows
                if self._is_finite_number(row.get(f"{leg}_swing_height_m"))
            ]
            if swing_values:
                add_metric(f"{leg}_swing_height_p95", self._percentile(swing_values, 0.95), "m")
                add_metric(f"{leg}_swing_height_max", max(swing_values), "m")
            else:
                add_metric(f"{leg}_swing_height_p95", float("nan"), "m")
                add_metric(f"{leg}_swing_height_max", float("nan"), "m")

        return summary_rows

    def _write_samples_csv(self, rows: list[dict[str, object]]) -> None:
        with self.samples_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary_csv(self, summary_rows: list[dict[str, object]]) -> None:
        with self.summary_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(summary_rows)

    def _write_plot(self, rows: list[dict[str, object]], z_ref_by_leg: dict[str, float]) -> None:
        if not rows:
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            print("[INFO] Skipping foot-tip trajectory plot because matplotlib is not installed.")
            return

        leg_colors = {
            "FL": "#1f77b4",
            "FR": "#ff7f0e",
            "RL": "#2ca02c",
            "RR": "#9467bd",
        }

        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=160)

        for leg in self.LEG_NAMES:
            x_b_vals = [float(row[f"{leg}_x_b_m"]) for row in rows]
            z_b_vals = [float(row[f"{leg}_z_b_m"]) for row in rows]
            contact = [int(row[f"{leg}_contact"]) for row in rows]

            step_segments = self._extract_step_segments_body(x_b_vals, z_b_vals, contact)
            if not step_segments:
                # Fallback: if no complete lift-off -> touchdown cycle is detected, keep a faint raw trace.
                x0 = x_b_vals[0] if x_b_vals else 0.0
                z0 = z_b_vals[0] if z_b_vals else 0.0
                raw_dx = [abs(x - x0) for x in x_b_vals]
                raw_dz = [z - z0 for z in z_b_vals]
                ax.plot(raw_dx, raw_dz, color=leg_colors[leg], linewidth=0.9, alpha=0.35)
                continue

            for seg_x, seg_y in step_segments:
                ax.plot(seg_x, seg_y, color=leg_colors[leg], linewidth=0.9, alpha=0.30)

            mean_x, mean_y = self._mean_segment_curve(step_segments, num_points=60)
            if mean_x and mean_y:
                ax.plot(mean_x, mean_y, color=leg_colors[leg], linewidth=2.0, alpha=0.95)

        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Step length magnitude within step |x_b(td) - x_b(lo)| (m)")
        ax.set_ylabel("Foot height relative to local lift-off baseline (Δz_b, m)")
        ax.set_title("Foot trajectories per step (body frame, positive step length)")
        ax.grid(axis="both", linestyle=":", linewidth=0.8, alpha=0.45)

        legend_handles = [
            Line2D([0], [0], color=leg_colors[leg], linewidth=2.0, label=leg) for leg in self.LEG_NAMES
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, title="Leg")

        step_counts = []
        for leg in self.LEG_NAMES:
            contact = [int(row[f"{leg}_contact"]) for row in rows]
            step_counts.append(f"{leg}:{self._count_completed_steps(contact)}")
        fig.text(0.01, 0.965, "completed steps: " + " | ".join(step_counts), fontsize=8)

        summary_line = " | ".join([f"{leg} z_ref={z_ref_by_leg.get(leg, 0.0):.3f} m" for leg in self.LEG_NAMES])
        fig.text(0.01, 0.01, f"global stance z_ref (base frame): {summary_line}", fontsize=8)

        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        fig.savefig(self.plot_path)

        try:
            plt.show()
        except Exception as err:
            print(f"[INFO] Unable to display foot-tip trajectory plot window: {err}")
        finally:
            plt.close(fig)

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

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

    def _resolve_foot_indices(
        self,
        base_env,
        body_pos_w: torch.Tensor,
        net_forces_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Resolve FL/FR/RL/RR index ordering for kinematics and contact force tensors."""
        target_names = [f"{leg}_foot" for leg in self.LEG_NAMES]

        body_ids = self._foot_body_ids
        if body_ids is None or body_ids.numel() < 4 or int(torch.max(body_ids).item()) >= body_pos_w.shape[1]:
            body_ids = None
            robot = getattr(base_env, "_robot", None)
            if robot is not None and hasattr(robot, "find_bodies"):
                try:
                    ids, names = robot.find_bodies(target_names)
                    body_ids = self._ids_from_names(ids, names, target_names, device=body_pos_w.device)
                except Exception:
                    body_ids = None
            if body_ids is None:
                fallback_ids = getattr(base_env, "_feet_ids", None)
                if fallback_ids is not None:
                    body_ids = self._to_long_tensor(fallback_ids, device=body_pos_w.device)
            if body_ids is not None and body_ids.numel() >= 4:
                body_ids = body_ids[:4]
            self._foot_body_ids = body_ids

        contact_ids = self._foot_contact_ids
        if isinstance(net_forces_w, torch.Tensor):
            max_contact_idx = net_forces_w.shape[1]
            if (
                contact_ids is None
                or contact_ids.numel() < 4
                or int(torch.max(contact_ids).item()) >= max_contact_idx
            ):
                contact_ids = None
                contact_sensor = getattr(base_env, "_contact_sensor", None)
                if contact_sensor is not None and hasattr(contact_sensor, "find_bodies"):
                    try:
                        ids, names = contact_sensor.find_bodies(target_names)
                        contact_ids = self._ids_from_names(ids, names, target_names, device=net_forces_w.device)
                    except Exception:
                        contact_ids = None
                if contact_ids is None:
                    fallback_ids = getattr(base_env, "_feet_ids", None)
                    if fallback_ids is not None:
                        contact_ids = self._to_long_tensor(fallback_ids, device=net_forces_w.device)
                if contact_ids is not None and contact_ids.numel() >= 4:
                    contact_ids = contact_ids[:4]
                self._foot_contact_ids = contact_ids
        else:
            contact_ids = None

        return body_ids, contact_ids

    @staticmethod
    def _to_long_tensor(values, device: torch.device) -> torch.Tensor | None:
        if isinstance(values, torch.Tensor):
            return values.detach().to(device=device, dtype=torch.long).view(-1)
        try:
            return torch.tensor(values, device=device, dtype=torch.long).view(-1)
        except Exception:
            return None

    def _ids_from_names(
        self,
        ids,
        names,
        target_names: list[str],
        device: torch.device,
    ) -> torch.Tensor | None:
        id_tensor = self._to_long_tensor(ids, device=device)
        if id_tensor is None or id_tensor.numel() == 0:
            return None

        if not isinstance(names, (list, tuple)) or len(names) != int(id_tensor.numel()):
            return id_tensor

        name_list = [str(name) for name in names]
        selected: list[int] = []
        used_indices: set[int] = set()
        for target in target_names:
            target_idx = None
            for idx, body_name in enumerate(name_list):
                if idx in used_indices:
                    continue
                if target in body_name:
                    target_idx = idx
                    break
            if target_idx is None:
                return id_tensor
            used_indices.add(target_idx)
            selected.append(int(id_tensor[target_idx].item()))

        return torch.tensor(selected, device=device, dtype=torch.long)

    @staticmethod
    def _extract_step_segments_body(
        x_b_vals: list[float],
        z_b_vals: list[float],
        contact: list[int],
    ) -> list[tuple[list[float], list[float]]]:
        """Return swing segments [lift-off, touchdown] with positive x and per-step local y baseline."""
        if not x_b_vals or not z_b_vals or not contact:
            return []
        if len(x_b_vals) != len(z_b_vals) or len(x_b_vals) != len(contact):
            return []

        segments: list[tuple[list[float], list[float]]] = []
        swing_start_idx: int | None = 0 if int(contact[0]) == 0 else None
        prev_contact = int(contact[0])

        for idx in range(1, len(contact)):
            curr_contact = int(contact[idx])
            # Lift-off: contact -> swing
            if prev_contact == 1 and curr_contact == 0:
                swing_start_idx = idx
            # Touchdown: swing -> contact
            if prev_contact == 0 and curr_contact == 1 and swing_start_idx is not None:
                seg_indices = list(range(swing_start_idx, idx + 1))
                if len(seg_indices) >= 3:
                    x0 = x_b_vals[swing_start_idx]
                    x_td = x_b_vals[idx]
                    step_len_mag = abs(x_td - x0)
                    z0 = z_b_vals[swing_start_idx]
                    seg_count = len(seg_indices)
                    seg_x = [
                        (step_len_mag * local_i / float(seg_count - 1)) if seg_count > 1 else 0.0
                        for local_i in range(seg_count)
                    ]
                    seg_y = [z_b_vals[j] - z0 for j in seg_indices]
                    segments.append((seg_x, seg_y))
                swing_start_idx = None
            prev_contact = curr_contact

        return segments

    @staticmethod
    def _mean_segment_curve(
        segments: list[tuple[list[float], list[float]]],
        num_points: int = 60,
    ) -> tuple[list[float], list[float]]:
        """Compute mean per-step trajectory over normalized phase to make subtle legs visible."""
        if not segments or num_points < 2:
            return [], []

        phase_grid = [idx / float(num_points - 1) for idx in range(num_points)]
        x_acc = [0.0] * num_points
        y_acc = [0.0] * num_points
        valid_count = 0

        for seg_x, seg_y in segments:
            n = min(len(seg_x), len(seg_y))
            if n < 2:
                continue
            src_phase = [idx / float(n - 1) for idx in range(n)]
            interp_x = PlayFootTipRecorder._interp_linear(src_phase, seg_x[:n], phase_grid)
            interp_y = PlayFootTipRecorder._interp_linear(src_phase, seg_y[:n], phase_grid)
            for i in range(num_points):
                x_acc[i] += interp_x[i]
                y_acc[i] += interp_y[i]
            valid_count += 1

        if valid_count == 0:
            return [], []
        mean_x = [value / valid_count for value in x_acc]
        mean_y = [value / valid_count for value in y_acc]
        return mean_x, mean_y

    @staticmethod
    def _interp_linear(x_src: list[float], y_src: list[float], x_dst: list[float]) -> list[float]:
        """Piecewise-linear interpolation without numpy dependency."""
        if len(x_src) != len(y_src) or len(x_src) < 2:
            return [y_src[0] if y_src else 0.0 for _ in x_dst]

        result: list[float] = []
        j = 0
        for xq in x_dst:
            while j < len(x_src) - 2 and x_src[j + 1] < xq:
                j += 1
            x0, x1 = x_src[j], x_src[j + 1]
            y0, y1 = y_src[j], y_src[j + 1]
            if x1 <= x0:
                result.append(y0)
                continue
            ratio = (xq - x0) / (x1 - x0)
            ratio = max(0.0, min(1.0, ratio))
            result.append(y0 * (1.0 - ratio) + y1 * ratio)
        return result

    @staticmethod
    def _count_completed_steps(contact: list[int]) -> int:
        """Count completed swing cycles (lift-off followed by touchdown)."""
        if not contact:
            return 0
        count = 0
        in_swing = int(contact[0]) == 0
        for idx in range(1, len(contact)):
            prev_c = int(contact[idx - 1])
            curr_c = int(contact[idx])
            if prev_c == 1 and curr_c == 0:
                in_swing = True
            elif prev_c == 0 and curr_c == 1 and in_swing:
                count += 1
                in_swing = False
        return count
