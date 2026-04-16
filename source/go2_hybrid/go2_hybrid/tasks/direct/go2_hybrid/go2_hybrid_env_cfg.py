from pathlib import Path
import isaaclab.envs.mdp as mdp
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import NoiseModelCfg, GaussianNoiseCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import observation_terms as obs_terms

from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfInvertedPyramidStairsTerrainCfg, MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

@configclass
class EventCfg:
    """Domain randomization events for DirectEnv."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 2.5),
            "dynamic_friction_range": (0.2, 2.3),
            "restitution_range": (0.0, 0.8),
            "num_buckets": 64,
        },
    )

    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
            },
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-1.0, 1.0),
        },
        interval_range_s=(3.0, 10.0),
        is_global_time=False,
    )




@configclass
class ObservationNoiseCfg:
    """ObsTerm-style observation corruption settings for DirectRLEnv policy inputs."""

    enable_corruption: bool = True

    root_lin_vel_b = ObsTerm(
        func=obs_terms.root_lin_vel_b,
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    root_ang_vel_b = ObsTerm(
        func=obs_terms.root_ang_vel_b,
        noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity_b = ObsTerm(
        func=obs_terms.projected_gravity_b,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos = ObsTerm(
        func=obs_terms.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=obs_terms.joint_vel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
    )


@configclass
class Go2HybridEnvCfg(DirectRLEnvCfg):

    episode_length_s = 100.0
    decimation = 4
    action_space = 12          # Unitree Go2 typically 12 actuated joints
    observation_space = 184    # will be validated at runtime
    state_space = 314
    dt=0.005
    action_scale = 0.25
    max_episode_length = int(episode_length_s / (dt * decimation))
    base_x_offset: float = 2.6
    base_z_offset: float = 0.4
    end_point_pos: float = 18.0

    lidar_range: float = 70.0
    # Hook domain randomization terms into DirectRLEnv/EventManager.
    events: EventCfg = EventCfg()
    # ObsTerm-style per-signal noise configuration for policy observations.
    obs_noise: ObservationNoiseCfg = ObservationNoiseCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
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
        terrain_type="usd",
        usd_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/updown_18cm_wide.usdz",
        # fixed terrain for this branch
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
        update_period = 1.0/5.5, # 5.5 Hz
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
