# go2_hybrid_env_cfg.py
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils

from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfInvertedPyramidStairsTerrainCfg, MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


@configclass
class Go2HybridEnvCfg(DirectRLEnvCfg):

    episode_length_s = 20.0
    decimation = 4
    action_space = 12          # Unitree Go2 typically 12 actuated joints
    observation_space = 48     # will be validated at runtime
    state_space = 0

    #simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ---------- scene ----------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.0, replicate_physics=True
    )

    
    #-----terrain-------NOT USED--#

    terrain_gen= TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=5.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            # "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            #     proportion=0.2,
            #     step_height_range=(0.05, 0.23),
            #     step_width=0.3,
            #     platform_width=3.0,
            #     border_width=1.0,
            #     holes=False,
            # ),
        },
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/Terrain",
        #terrain_type="generator",
        #terrain_generator=terrain_gen,
        terrain_type="plane",
        #usd_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/double_stairs_colour.usdz",
        
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ---------- robot ----------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            #usd_path="/home/ril/go2_hybrid/go2_hybrid/source/go2_hybrid/assets/go2_hybrid/go2_hybrid_model.usd",
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
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                # "F[L,R]_thigh_joint": 0.4,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
        },
    )

    #-------sensor---------#
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/.*",  # <- exact subtree
    history_length=3,
    update_period=0.005,
    track_air_time=True,
    )

    height_scanner=RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=40,
            vertical_fov_range= [-7,52], horizontal_fov_range=[-90,90], horizontal_res=1.0
        ),
        mesh_prim_paths=["/World/Terrain"],
        update_period=0.1,
        history_length=0,
        debug_vis=False,
    )
    

    # ---------- reward scales ----------
    #   self.rewards.track_lin_vel_xy_exp.weight  -> lin_vel_reward_scale
    #   self.rewards.track_ang_vel_z_exp.weight  -> yaw_rate_reward_scale
    #   self.rewards.dof_torques_l2.weight       -> joint_torque_reward_scale
    #   self.rewards.dof_acc_l2.weight           -> joint_accel_reward_scale
    #   self.rewards.feet_air_time.weight        -> feet_air_time_reward_scale
    #   self.rewards.undesired_contacts=None     -> undesired_contact_reward_scale = 0.0

    lin_vel_reward_scale: float = 5.0
    yaw_rate_reward_scale: float = 0.75
    joint_torque_reward_scale: float = -2.0e-4
    joint_accel_reward_scale: float = -2.5e-7
    feet_air_time_reward_scale: float = 0.01
    undesired_contact_reward_scale: float = 0.0
    # base_height_reward_scale: float = 1.0
    # base_height_below_penalty_scale: float = -5.0
    z_vel_reward_scale: float = -2.0
    ang_vel_reward_scale: float = -0.05
    action_rate_reward_scale: float = -0.01
    flat_orientation_reward_scale: float = -5.0

