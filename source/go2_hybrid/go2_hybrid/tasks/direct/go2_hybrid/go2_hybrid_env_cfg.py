from pathlib import Path
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import NoiseModelCfg, GaussianNoiseCfg


from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfInvertedPyramidStairsTerrainCfg, MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from .curriculum_phases import get_phase  # or from .phases import get_phase

@configclass
class Go2HybridEnvCfg(DirectRLEnvCfg):

    episode_length_s = 20.0
    decimation = 4
    action_space = 12          # Unitree Go2 typically 12 actuated joints
    observation_space = 184    # will be validated at runtime
    state_space = 314
    dt=0.005
    action_scale = 0.25
    max_episode_length = int(episode_length_s / (dt * decimation))

    ###### Phase related configs ######
    phase_id: int = 3 #manually change this for different curriculum phase
    end_point_pos: float = 0.0 #set in post __init__ below
    base_x_offset: float = 0.0
    base_z_offset: float = 0.0

    reward_module: str = "rewards_UD_icra_p2"
    terrain_path: str | None = None
    ######

    lidar_range: float = 70.0

    #simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0, #physics timestep 
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ---------- scene ----------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200, env_spacing=0.0, replicate_physics=True
    )

    if phase_id != 0:
        # Action noise (applied to the raw [-1, 1] actions coming from the policy)
        action_noise_model = NoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(
                mean=0.0,
                std=0.20,          
                operation="add",
            )
        )

    terrain = TerrainImporterCfg(
        prim_path="/World/Terrain",
        #terrain_type="generator",
        #terrain_generator=terrain_gen,
        terrain_type="",
        usd_path="",        
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ---------- robot ----------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_normal/go2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            
            # "base_legs": DCMotorCfg(
            #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            #     effort_limit=23.5,
            #     saturation_effort=23.5,
            #     velocity_limit=30.0,
            #     stiffness=25.0,
            #     damping=0.5, 
            #     friction=0.0,
            # ),
            "base_legs": DelayedPDActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                # saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
                min_delay=0, #physics timesteps
                max_delay=3, #physics timesteps (5ms)
            ),
        },
    )

    #-------sensor---------#
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/.*",  # <- exact subtree
    history_length=3,
    update_period=0.005,
    track_air_time=True,
    debug_vis=False,
    filter_prim_paths_expr="/World/Terrain",
    max_contact_data_count_per_prim=20,
    )

    lidar_scanner=RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Head_lower",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=5,
            vertical_fov_range= [-60,-20], horizontal_fov_range=[-45,45], horizontal_res=10.0        ),
        mesh_prim_paths=["/World/Terrain"],
        update_period = 1.0 / 5.5,
        history_length=0,
        debug_vis=True,
    )
    
    height_scanner=RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="base",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6,0.4]),    
        mesh_prim_paths=["/World/Terrain"],
        debug_vis=False,
    )

    def __post_init__(self):
        """Resolve phase-dependent config from curriculum_phases.py."""
        try:
            super().__post_init__()
        except AttributeError:
            pass

        phase_spec = get_phase(int(self.phase_id))

        # Always-available fields
        self.reward_module = phase_spec.reward_module
        self.terrain_path = phase_spec.terrain_path
        self.episode_length_s = phase_spec.episode_length_s

        if phase_spec.end_point_pos is not None:
            self.end_point_pos = float(phase_spec.end_point_pos)
        if phase_spec.base_x_offset is not None:
            self.base_x_offset = float(phase_spec.base_x_offset)
        if phase_spec.base_z_offset is not None:
            self.base_z_offset = float(phase_spec.base_z_offset)

        if phase_spec.terrain_path is not None:
            self.terrain.terrain_type = "usd"
            self.terrain.usd_path = phase_spec.terrain_path
        else:
            # Phase 0: ensure /World/Terrain exists as a mesh for raycasters/contact filtering
            self.terrain.terrain_type = "plane"
