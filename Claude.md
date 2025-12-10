# **Project: PPO-Based Stair Traversal for Unitree Go2 Using IsaacSim 5.0.0 + IsaacLab 2.2.0**

This project implements a **DirectRLEnv-based reinforcement learning pipeline** to train a **Unitree Go2** quadruped robot to **ascend (and later descend) custom stair terrains** using PPO.
A **LiDAR ray-based terrain perception module** guides the policy, producing a **non-blind, perception-aware controller**.

The project follows a **three-phase curriculum**, and all implementation details—including environment logic, reward shaping, termination functions, and LiDAR processing—are included below.

---

# **1. System Overview**

### **Simulation Stack**

* **IsaacSim 5.0.0**
* **IsaacLab 2.2.0**
* Custom project built using the *IsaacLab External Project Generator*:
  [https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html)

### **Goal**

Train a robust locomotion policy that:

1. **Ascends stairs** of various geometries (height, width, #steps).
2. Uses **LiDAR-based height estimation** to shape stable, terrain-aware locomotion.
3. Optionally **descends** stairs using continued curriculum fine-tuning.
4. Explores research questions about **reward shaping** and **observation design**, which form the core contribution of the master’s thesis.

---

# **2. Curriculum Learning Framework**

The training uses a **three-phase curriculum**:

### **Phase 0 — Flat Ground Locomotion**

* Learn stable trotting.
* Use **left–right symmetry (augmentation)** across sagittal plane.
* Stable forward locomotion and orientation control.

### **Phase 1 — Stair Ascending**

* Load best checkpoint from Phase 0.
* Terrain replaced with a custom staircase mesh (`double_stairs_10_colour.usdz`).
* Introduce **terrain-adaptive rewards** (e.g., LiDAR-based base height tracking, foot clearance, rear-foot stepping timing).
* Remove symmetry augmentation, because stair climbing is inherently asymmetric.

### **Phase 2 — Stair Ascend + Descend**

* Fine-tune from best ascending controller.
* Learn bidirectional stability and terrain negotiation.

---

# **3. Project Structure**

```
go2_hybrid/
  source/go2_hybrid/go2_hybrid/tasks/direct/go2_hybrid/
    assets/
      go2_hybrid/double_stairs_10_colour.usdz
      go2_normal/go2.usd                    # Unitree Go2 model (12 DOF)
    agents/
      rsl_rl_ppo_cfg.yaml                   # hyperparameters
    go2_hybrid_env_cfg.py                   # environment config
    go2_hybrid_env.py                       # main DirectRLEnv implementation
    rewards.py                              # reward shaping
    terminations.py                         # termination conditions
    __init__.py
  scripts/rsl_rl/
    train.py                                # PPO training script
    play.py                                 # visualization script
    cli_args.py
```

Key source files (with citations):

* Environment logic: **go2_hybrid_env.py** 
* Environment configuration: **go2_hybrid_env_cfg.py** 
* Reward functions: **rewards.py** 
* Termination conditions: **terminations.py** 

---

# **4. Environment Description**

The main environment is implemented in **Go2HybridEnv** (DirectRLEnv) .

### **Robot Model**

* USD: `go2.usd`
* Degrees of freedom: **12 actuated joints** (4 hips, 4 thighs, 4 calves).

### **Terrain**

* Stairs: `double_stairs_10_colour.usdz`
* Trapezium staircase with **6 steps**, each approx **10 cm**.

### **Sensors**

1. **LiDAR RayCaster**

   * Mounted on head
   * 5 vertical channels
   * Horizontal FOV ±45°, vertical FOV −60° to −20°
   * Provides environment height information for reward shaping and observations.

2. **Height scanner**

   * Downward rays from base
   * Provides relative elevation for base-height reward.

3. **Contact Sensor**

   * Tracks forces, air times, sticking conditions.

### **Observations**

Policy observation includes:

* Base linear/angular velocity (body frame)
* Projected gravity
* Commanded velocity + heading
* Joint position errors + velocities
* Previous actions
* **Flattened LiDAR hits in base frame**
* Height-scanner–based elevation observations

Full observation dimension: **184**.
Critic receives **privileged** information: body forces, torques, contact history, ground height, etc. (state space = 314).

---

# **5. LiDAR Processing Pipeline**

Located in **go2_hybrid_env.py** and **rewards.py**.
All LiDAR hits are:

1. Retrieved in **world frame**
2. Transformed to **base frame** using quaternion inverse
3. Normalized to range [0,1]
4. Flattened for policy input

Terrain height is estimated using the **lowest-elevation LiDAR channel**, averaging z-values in base frame:
`terrain_height_b = mean(z_vals)`
(see **get_height_lidar** in rewards.py) 

---

# **6. Reward Function Design**

Reward components are defined in **rewards.py** .

### **Velocity Tracking**

* **track_modified_vel_reward** blends:

  * base-frame tracking
  * world-frame forward progress
  * interpolation based on pitch angle (smooth transition for stairs).

### **Terrain-Aware Rewards**

* **foot_clearance_reward**
  Encourages adequate clearance above LiDAR-estimated terrain height.

* **base_height_l2_lidar**
  Penalizes deviation from (terrain + target offset).

* **rear_match_front**
  Encourages hind legs to step into previously stable footholds created by front legs.

* **foot_vertical_accel_reward**
  Encourages upward recovery motion when a foot is stuck.

### **Stability Penalties**

* Orientation penalty (flat_orientation)
* Angular velocity penalty
* Joint torque, acceleration, jerk (smoothness_penalty)
* Undesired contacts (thigh, head, calves)
* Backward velocity penalty (important for stairs)

### **Phase-specific emphasis**

During stair training, weightings favour:

* clearance
* base height tracking
* rear-leg stepping timing
* forward world motion

All rewards combined via:
`total = Σ (w_i * reward_i * dt)`

---

# **7. Termination Conditions**

Defined in **terminations.py** .

Key terminations:

* **time_out** — episode end
* **illegal_contact** — large force on base (fall or collision)
* **out_of_bounds** — falling below terrain
* **flipped_over** — projected gravity indicates rollover
* **stuck** — commanded forward but no progress for 2 seconds

The **stuck** logic maintains an env-level counter updated every simulation step.

---

# **8. Command Sampling & Action Scaling**

Inside `Go2HybridEnv._pre_physics_step` and `resample_commands()`:

* Commands resampled every 8–12 seconds.
* For stairs (Phase 1):

  * Heading fixed toward stairs.
  * Forward vx ∈ [0.4, 1.0]
  * vy = 0, yaw_rate = 0.

Actions are scaled:
`processed_actions = action_scale * actions + default_joint_pos`,
where `action_scale = 0.25`.

---

# **9. Asymmetric Actor–Critic Architecture**

Actor receives **partial non-privileged** observations.
Critic receives **full privileged state**.
Implemented implicitly via `"policy"` and `"critic"` keys returned by `_get_observations()`.

This improves stability on rough terrains and helps the critic learn meaningful value estimations on stairs.

---

# **10. Training Setup (PPO)**

Located in `agents/rsl_rl_ppo_cfg.yaml` (not included in this summary).

Typical important PPO settings:

* Large batch size (due to 200 parallel envs)
* Long horizon (20s episode length)
* Adaptive KL and entropy regularization
* Higher learning rate during early curriculum phases
* Strong reward scaling for terrain-critical terms (clearance, base-height)

---

# **11. Research Focus for Thesis**

Your thesis contribution is mainly in:

### **Reward Design**

* LiDAR-based adaptive clearance.
* Terrain-relative base-height constraint.
* Rear-vs-front stepping alignment heuristic.
* Stuck-recovery shaping.

### **Observation Engineering**

* Multi-channel LiDAR flattening pipeline.
* Privileged critic state.
* Removal vs. inclusion of symmetry augmentation across terrain types.

### **Effect of Curriculum**

* Influence of pretraining on flat vs direct stair training.
* Behaviour transfer across ascending/descending tasks.

---

# **12. Future Extensions**

Potential thesis extensions:

* Learning **descending-optimized gait** (different dynamics on down-slope).
* Incorporating **memory (GRU)** to improve terrain anticipation.
* Using **self-supervised height prediction** (instead of raw LiDAR).
* Adding **collision-free foothold selection** as auxiliary tasks.
* Investigating **latent sim-to-real robustness**.

---

# **13. Summary**

This repository implements a complete reinforcement-learning locomotion system using IsaacSim + IsaacLab, featuring:

* High-fidelity LiDAR-based observation design
* Reward shaping focused on perception-aware stair traversal
* Curriculum learning across flat → ascend → descend
* Strong termination logic
* Asymmetric actor-critic PPO