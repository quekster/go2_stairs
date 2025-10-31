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
    observation_space = 1398    # will be validated at runtime 21648 
    state_space = 0
    dt=0.005

    action_scale = 0.25
    max_episode_length = int(episode_length_s / (dt * decimation))

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
        num_envs=200, env_spacing=1.0, replicate_physics=True
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
            pos=(0.0, 0.0, 0.2),
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
    debug_vis=False,
    filter_prim_paths_expr="/World/Terrain",
    max_contact_data_count_per_prim=20,
    )

    height_scanner=RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Head_upper",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=5,
            vertical_fov_range= [-50,-10], horizontal_fov_range=[-45,45], horizontal_res=2.0
        ),
        mesh_prim_paths=["/World/Terrain"],
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )
    

