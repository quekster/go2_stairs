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


# --- DR DISABLED (EVENT CFG): full class kept here but commented out ---
# @configclass
# class EventCfg:
#     """Domain randomization events for DirectEnv."""
#
#     robot_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "static_friction_range": (0.2, 2.5),
#             "dynamic_friction_range": (0.2, 2.3),
#             "restitution_range": (0.0, 0.8),
#             "num_buckets": 64,
#         },
#     )
#
#     robot_actuator_gains = EventTerm(
#         func=mdp.randomize_actuator_gains,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
#             "stiffness_distribution_params": (0.85, 1.15),
#             "damping_distribution_params": (0.85, 1.15),
#             "operation": "scale",
#             "distribution": "uniform",
#         },
#     )
#
#     add_base_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="base"),
#             "mass_distribution_params": (-5.0, 5.0),
#             "operation": "add",
#         },
#     )
#
#     base_com = EventTerm(
#         func=mdp.randomize_rigid_body_com,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="base"),
#             "com_range": {
#                 "x": (-0.05, 0.05),
#                 "y": (-0.05, 0.05),
#                 "z": (-0.01, 0.01),
#             },
#         },
#     )
#
#     base_external_force_torque = EventTerm(
#         func=mdp.apply_external_force_torque,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="base"),
#             "force_range": (-5.0, 5.0),
#             "torque_range": (-1.0, 1.0),
#         },
#         interval_range_s=(10.0, 10.0),
#         is_global_time=False,
#     )


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
    phase_id: int = 0 #manually change this for different curriculum phase
    end_point_pos: float = 0.0 #set in post __init__ below
    base_x_offset: float = 0.0
    base_z_offset: float = 0.0

    reward_module: str = "rewards_UD_icra_p2"
    terrain_path: str | None = None
    ######

    lidar_range: float = 70.0
    ground_plane_height: float = -0.0025
    ground_plane_size: tuple[float, float] = (2.0e6, 2.0e6)
    ground_plane: AssetBaseCfg | None = None
    ground_contact_sensor_cfg: ContactSensorCfg | None = None

    # ------- domain randomization (DirectEnv startup/reset-time) ------- #
    # --- DR DISABLED (FRICTION): friction/material randomization ---
    # randomize_rigid_body_material: bool = True
    randomize_rigid_body_material: bool = False
    static_friction_range: tuple[float, float] = (0.2, 2.5)
    dynamic_friction_range: tuple[float, float] = (0.2, 2.3)
    restitution_range: tuple[float, float] = (0.0, 0.8)
    material_num_buckets: int = 64

    # --- DR DISABLED (PD TORQUE): actuator gain randomization ---
    # randomize_actuator_gains: bool = True
    randomize_actuator_gains: bool = False
    stiffness_distribution_params: tuple[float, float] = (0.85, 1.15)
    damping_distribution_params: tuple[float, float] = (0.85, 1.15)

    randomize_add_base_mass: bool = True
    add_base_mass_distribution_params: tuple[float, float] = (-5.0, 5.0)

    randomize_base_com: bool = True
    base_com_x_range: tuple[float, float] = (-0.05, 0.05)
    base_com_y_range: tuple[float, float] = (-0.05, 0.05)
    base_com_z_range: tuple[float, float] = (-0.01, 0.01)

    # --- DR DISABLED (EXTERNAL FORCE): external force/torque randomization ---
    # randomize_base_external_force_torque: bool = True
    randomize_base_external_force_torque: bool = False
    base_external_force_range: tuple[float, float] = (-5.0, 5.0)
    base_external_torque_range: tuple[float, float] = (-1.0, 1.0)
    base_external_force_mode: str = "interval"  # "reset" or "interval"
    base_external_force_interval_s: tuple[float, float] = (3.0, 10.0)
    base_external_force_global_time: bool = False

    # #simulation for phase 3-5
    # sim: SimulationCfg = SimulationCfg(
    #     dt=1.0 / 200.0, #physics timestep 
    #     render_interval=decimation,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="average",
    #         restitution_combine_mode="average",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    # )

    #simulation for phase 0-2
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

    # --- DR DISABLED (EVENT CFG): disable all event-based DR terms globally ---
    # events: EventCfg = EventCfg()
    events = None

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
        #Phase 3 to 5 
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="average",
        #     restitution_combine_mode="average",
        #     static_friction=0.8,
        #     dynamic_friction=0.7,
        #     restitution=0.0,
        # ),

        #Phase 0-2
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
            
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5, 
                friction=0.0,
            ),
            # "base_legs": DelayedPDActuatorCfg(
            #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            #     effort_limit=23.5,
            #     # saturation_effort=23.5,
            #     velocity_limit=30.0,
            #     stiffness=25.0,
            #     damping=0.5,
            #     friction=0.0,
            #     min_delay=0, #physics timesteps
            #     max_delay=3, #physics timesteps (5ms)
            # ),
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
        update_period = 0.0,
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

        # if self.events is not None:
        #     if self.randomize_rigid_body_material:
        #         material_event = self.events.robot_physics_material
        #         material_event.params["static_friction_range"] = (0.2, 2.5)
        #         material_event.params["dynamic_friction_range"] = (0.2, 2.3)
        #         material_event.params["restitution_range"] = (0.0, 0.8)
        #         material_event.params["num_buckets"] = 64
        #     else:
        #         self.events.robot_physics_material = None

        #     if self.randomize_actuator_gains:
        #         gains_event = self.events.robot_actuator_gains
        #         gains_event.params["stiffness_distribution_params"] = (0.85, 1.15)
        #         gains_event.params["damping_distribution_params"] = (0.85, 1.15)
        #     else:
        #         self.events.robot_actuator_gains = None

        #     if self.randomize_add_base_mass:
        #         base_mass_event = self.events.add_base_mass
        #         base_mass_event.params["mass_distribution_params"] = self.add_base_mass_distribution_params
        #     else:
        #         self.events.add_base_mass = None

        #     if self.randomize_base_com:
        #         base_com_event = self.events.base_com
        #         base_com_event.params["com_range"] = {
        #             "x": self.base_com_x_range,
        #             "y": self.base_com_y_range,
        #             "z": self.base_com_z_range,
        #         }
        #     else:
        #         self.events.base_com = None

        #     if self.randomize_base_external_force_torque:
        #         ext_wrench_event = self.events.base_external_force_torque
        #         ext_wrench_event.params["force_range"] = self.base_external_force_range
        #         ext_wrench_event.params["torque_range"] = self.base_external_torque_range
        #         if self.base_external_force_mode not in ("reset", "interval"):
        #             raise ValueError(
        #                 f"Unsupported base_external_force_mode='{self.base_external_force_mode}'. "
        #                 "Use 'reset' or 'interval'."
        #             )
        #         ext_wrench_event.mode = self.base_external_force_mode
        #         ext_wrench_event.interval_range_s = self.base_external_force_interval_s
        #         ext_wrench_event.is_global_time = self.base_external_force_global_time
        #     else:
        #         self.events.base_external_force_torque = None

        # Phase 4 only: add a global fallback ground plane below the terrain.
        if int(self.phase_id) == 4:
            self.ground_plane = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=self.terrain.physics_material,
                    size=self.ground_plane_size,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, self.ground_plane_height)),
                collision_group=-1,
            )
            self.ground_contact_sensor_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                history_length=3,
                update_period=0.005,
                track_air_time=False,
                debug_vis=False,
                filter_prim_paths_expr=[f"{self.ground_plane.prim_path}/GroundPlane/CollisionPlane"],
                max_contact_data_count_per_prim=20,
            )
        else:
            self.ground_plane = None
            self.ground_contact_sensor_cfg = None
