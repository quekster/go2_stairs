from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CurricumlumPhase:
    """Declarative specification of what changes per phase."""
    phase_id: int
    name: str
    reward_module: str
    terrain_path: str | None = None
    notes: str | None = None
    episode_length_s: float = 20.0

    #spawn/end point coordinates for different phase terrains

    end_point_pos: Optional[float] = None
    base_x_offset: Optional[float] = None
    base_z_offset: Optional[float] = None

# ----------------------------
# Phase registry
# ----------------------------
PHASES: Dict[int, CurricumlumPhase] = {
    0: CurricumlumPhase(
        phase_id=0,
        name="plane_p0",
        reward_module="rewards_plane_p0",
        terrain_path=None,
        notes="Flat-ground walking with all cmd_vel + symmetry.",
        base_x_offset=2.5,
        base_z_offset= 0.4,
        episode_length_s= 20.0,
    ),
    1: CurricumlumPhase(
        phase_id=1,
        name="stairs_ascent_p1",
        reward_module="rewards_ascent_p1",
        terrain_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/100_stairs_10cm_ascending.usdz",
        notes="Ascent 100 stairs with 10cm height. Replaced plane rewards with uneven-terrain rewards.",
        end_point_pos= 18.0,
        base_x_offset= 2.5, 
        base_z_offset= 0.4,
        episode_length_s= 20.0,
    ),
    2: CurricumlumPhase(
        phase_id=2,
        name="stairs_updown_10cm_p2",
        reward_module="rewards_UD_p2_p3",
        terrain_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/updown_10cm.usdz",
        notes="Up+Down stairs 10cm. Continued training from ascent controller. Same rewards as icra_map",
        end_point_pos= 18.0,
        base_x_offset= 2.5, 
        base_z_offset= 0.4,
        episode_length_s= 20.0,
    ),
    3: CurricumlumPhase(
        phase_id=3,
        name="stairs_updown_18cm_p3",
        reward_module="rewards_UD_p2_p3",
        terrain_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/updown_18cm.usdz",
        notes="Up+Down stairs 18cm. Continued from 10cm curriculum.",
        end_point_pos= 18.0,
        base_x_offset= 2.5, 
        base_z_offset= 0.4,
        episode_length_s= 100.0,
    ),
    4: CurricumlumPhase(
    phase_id=4,
    name="stairs_updown_18cm_p4",
    reward_module="rewards_UD_icra_p4",
    terrain_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/updown_18cm.usdz",
    notes="Up+Down stairs 18cm with modification. Continued from previous 18cm curriculum.",
    end_point_pos= 18.0,
    base_x_offset= 2.5, 
    base_z_offset= 0.4,
    episode_length_s= 100.0,
    ),
    5: CurricumlumPhase(
        phase_id=5,
        name="icra_map_p5",
        reward_module="rewards_UD_icra_p4",
        terrain_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/icra_map_flat_long.usdz",  
        notes="ICRA challenge terrain. Reuses UD reward set.",
        end_point_pos= 33.0,
        base_x_offset= -1.0, 
        base_z_offset= 0.55,
        episode_length_s= 100.0,

    ),
}


def get_phase(phase_id: int) -> CurricumlumPhase:
    """Fetch phase spec; raises ValueError if unknown."""
    try:
        return PHASES[int(phase_id)]
    except Exception:
        valid = ", ".join(str(k) for k in sorted(PHASES.keys()))
        raise ValueError(f"Invalid phase_id={phase_id}. Valid phase_id: {valid}")