"""
Custom tools and utilities specific to ROBOT_TITLE robot.

This module contains helper functions, custom classes, and utilities
that are specific to working with the ROBOT_TITLE robot.
"""

import pandas as pd
import re
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from os.path import abspath
from scipy import signal  # Required for butter/filtfilt

from figaroh.tools.robot import Robot
from figaroh.identification.base_identification import BaseIdentification
from figaroh.tools.solver import LinearSolver

from figaroh.identification.identification_tools import get_standard_parameters
from figaroh.tools.regressor import (
    build_regressor_basic,
    get_index_eliminate,
    build_regressor_reduced,
    solve_differential_LMI_OLS
)
from figaroh.tools.qrdecomposition import get_baseParams

class H1v2Identification(BaseIdentification):
    """H1v2-specific dynamic parameter identification class."""
    
    P_BOUND = (0, None)    # Positive or zero (e.g., Mass, Friction, Actuator Inertia)
    N_BOUND = (None, None) # No bounds (e.g., Offsets, Center-of-Mass products, Cross-Inertia)

    def __init__(self, robot, config_file="config/h1v2_config.yaml"):
        """Initialize H1v2 identification with robot model and configuration."""
        super().__init__(robot, config_file)
        print("H1v2Identification initialized for H1v2 robot")
    
    def initialize_global(self, data_pathes_A, data_pathes_B, data_pathes_test, truncate_A=None, truncate_B=None, truncate_test=None):
        # Experiment A
        self.processed_data_A, self.num_samples_A = self.process_data(data_pathes=data_pathes_A, truncate=truncate_A)
        
        # Experiment B
        self.processed_data_B, self.num_samples_B = self.process_data(data_pathes=data_pathes_B, truncate=truncate_B)
        
        # Experiment test
        self.processed_data_test, self.num_samples_test = self.process_data(data_pathes=data_pathes_test, truncate=truncate_test)
        
    def process_data(self, data_pathes, truncate=None):
        """Load and process data"""
        filter_config = self.filter_config
        raw_data = self.load_trajectory_data(data_pathes)
        raw_data = self._truncate_data(raw_data, truncate)
        
        # 1. Filter Kinematics (Butterworth)
        processed_data, valid_segments = self.filter_kinematics_data(raw_data, filter_config)
        
        # 2. Filter Torques (Butterworth) using same segments
        processed_data["torques"] = self.process_torque_data(raw_data, processed_data, valid_segments)
        
        # 4. Build Full Config
        num_samples = processed_data["positions"].shape[0]
        processed_data = self._build_full_configuration(processed_data, num_samples)
                
        return processed_data, num_samples

    def calculate_full_regressor(self, model, processed_data, num_samples):
        """Build regressor matrix, compute pre-identified values of standard 
        parameters, compute joint torques based on pre-identified standard 
        parameters."""
        # Build full regressor matrix
        dynamic_regressor = build_regressor_basic(
            self.robot,
            processed_data["positions"],
            processed_data["velocities"],
            processed_data["accelerations"],
            self.identif_config,
        )
        
        # Compute standard parameters
        standard_parameter = get_standard_parameters(model, self.identif_config)

        # Convert all string values to floats in the standard_parameter dict
        for key, value in standard_parameter.items():
            if isinstance(value, str):
                standard_parameter[key] = float(value)

        # joint torque estimated from p,v,a with std params
        phi_ref = np.array(list(standard_parameter.values()))
        tau_ref = np.dot(dynamic_regressor, phi_ref)

        # filter only active joints
        tau_ref = tau_ref[range(len(self.identif_config["act_idxv"]) * num_samples)]
        
        return dynamic_regressor, standard_parameter, tau_ref
    
    def _build_full_configuration(self, processed_data, num_samples):
        """Build full configuration arrays."""
        required_keys = ["positions", "velocities", "accelerations"]
        for key in required_keys:
            if key not in processed_data or processed_data[key] is None:
                raise ValueError(f"Missing required data: {key}")

        q_active = processed_data["positions"]
        dq_active = processed_data["velocities"]
        ddq_active = processed_data["accelerations"]

        config_data = [
            (q_active, np.zeros_like(self.robot.q0), self.identif_config["act_idxq"]),
            (dq_active, np.zeros_like(self.robot.v0), self.identif_config["act_idxv"]),
            (ddq_active, np.zeros_like(self.robot.v0), self.identif_config["act_idxv"])
        ]

        full_configs = []
        for active_data, default_config, active_indices in config_data:
            full_config = np.tile(default_config, (num_samples, 1))
            full_config[:, active_indices] = active_data
            full_configs.append(full_config)

        config_keys = ["positions", "velocities", "accelerations"]
        processed_data.update(dict(zip(config_keys, full_configs)))
        
        return processed_data
        
    def load_trajectory_data(self, data_pathes):
        """Load and process CSV data."""
        ts = pd.read_csv(abspath(data_pathes["pos_data"]), usecols=[0]).to_numpy()
        pos = pd.read_csv(abspath(data_pathes["pos_data"]))
        vel = pd.read_csv(abspath(data_pathes["vel_data"]))
        eff = pd.read_csv(abspath(data_pathes["torque_data"]))

        cols = {"pos": [], "vel": [], "eff": []}
        for jn in self.identif_config["active_joints"]:
            cols["pos"].extend([col for col in pos.columns if jn in col])
            cols["vel"].extend([col for col in vel.columns if jn in col])
            cols["eff"].extend([col for col in eff.columns if jn in col])

        q = pos[cols["pos"]].to_numpy()
        dq = vel[cols["vel"]].to_numpy()
        tau = eff[cols["eff"]].to_numpy()
        
        return {
            "timestamps": ts,
            "positions": q,
            "velocities": dq,
            "accelerations": None,
            "torques": tau
        }

    def _apply_butterworth(self, data, dt):
        """
        Applies a zero-phase Butterworth Low-Pass filter.
        """
        if len(data) == 0:
            return data
            
        nyquist = 0.5 / dt
        # Safety check: Cutoff must be < Nyquist
        cutoff = self.filter_config["filter_params"]["cutoff_hz"]
        if cutoff >= nyquist:
            print(f"Warning: Cutoff {cutoff}Hz >= Nyquist {nyquist}Hz. Clamping.")
            cutoff = nyquist * 0.99
            
        norm_cutoff = cutoff / nyquist
        
        # Design Filter
        b, a = signal.butter(self.filter_config["filter_params"]["order"], norm_cutoff, 'low')
        
        # Apply Zero-Phase Filter (filtfilt)
        return signal.filtfilt(b, a, data, axis=0)
        
    def filter_kinematics_data(self, raw_data, filter_config=None):
        self.latest_filter_config = filter_config

        # --- 1. VALIDATION ---
        if raw_data.get("timestamps") is None: raise ValueError("Timestamps required")
        if raw_data.get("positions") is None: raise ValueError("Position data required")
        if raw_data.get("velocities") is None: raise ValueError("Velocity data required")

        raw_ts = np.array(raw_data["timestamps"]).flatten() 
        raw_pos = np.array(raw_data["positions"])
        raw_vel = np.array(raw_data["velocities"])
        
        dts = np.diff(raw_ts)
        median_dt = np.median(dts) if len(dts) > 0 else 0.001
        
        # --- 2. COMPUTE RAW DERIVATIVES (Finite Differences) ---
        raw_acc_calc = np.gradient(raw_vel, median_dt, axis=0)
        
        # --- 3. DETECTION PASS ---
        valid_mask = self._get_valid_mask_from_accel(raw_acc_calc, sensitivity=100.0, dilation=10)
        valid_segments = self._get_segment_slices(valid_mask)
        
        # --- 4. BUTTERWORTH FILTERING ---
        filtered_pos = self._apply_butterworth(raw_pos, median_dt)
        filtered_vel = self._apply_butterworth(raw_vel, median_dt)
        filtered_acc = np.gradient(filtered_vel, median_dt, axis=0)
        
        chunks = {"positions": [], "velocities": [], "accelerations": []}     

        for seg in valid_segments:
            chunks["positions"].append(filtered_pos[seg])
            chunks["velocities"].append(filtered_vel[seg])
            chunks["accelerations"].append(filtered_acc[seg])

        # --- 5. CONCATENATION ---
        processed_data = {}
        if chunks["positions"]:
            processed_data["positions"] = np.vstack(chunks["positions"])
            processed_data["velocities"] = np.vstack(chunks["velocities"])
            processed_data["accelerations"] = np.vstack(chunks["accelerations"])

            total_samples = len(processed_data["positions"])
            processed_data["timestamps"] = np.arange(total_samples) * median_dt + raw_ts[0]
        else:
            print("Warning: All data segments were rejected.")
            processed_data = {k: np.array([]) for k in ["positions", "velocities", "accelerations", "timestamps"]}

        # --- 6. PLOTTING ---
        # self._plot_segmentation_results(raw_ts, raw_pos, raw_vel, raw_acc_calc, processed_data, valid_segments, valid_mask)
        
        return processed_data, valid_segments

    def process_torque_data(self, raw_data, processed_data, valid_segments):
        """Process torque data using the same segments and filters."""

        raw_torques = np.array(raw_data["torques"])
        # Get delta for consistency
        raw_ts = np.array(raw_data["timestamps"]).flatten()
        dts = np.diff(raw_ts)
        median_dt = np.median(dts) if len(dts) > 0 else 0.001
        
        # --- BUTTERWORTH FILTERING ---
        filtered_torques = self._apply_butterworth(raw_torques, median_dt)

        torque_chunks = []

        for seg in valid_segments:
            seg_tau = filtered_torques[seg]
            torque_chunks.append(seg_tau)

        if torque_chunks:
            processed_data["torques"] = np.vstack(torque_chunks)
            # self._plot_single_signal(raw_data, valid_segments, "torques", raw_torques, processed_data["torques"])

            return processed_data["torques"]
        return np.array([])

    def _get_valid_mask_from_accel(self, acc, sensitivity=50.0, dilation=10):
        """Detects spikes using MAD and dilates by `dilation` frames."""
        if acc.ndim > 1:
            metric = np.max(np.abs(acc), axis=1)
        else:
            metric = np.abs(acc)
            
        median_val = np.median(metric)
        mad = np.median(np.abs(metric - median_val))
        if mad < 1e-6: mad = 1e-6
            
        modified_z_score = 0.6745 * (metric - median_val) / mad
        is_spike = modified_z_score > sensitivity
        
        # Dilation
        is_spike_dilated = np.copy(is_spike)
        for i in range(1, dilation + 1):
            is_spike_dilated[i:] |= is_spike[:-i]
            is_spike_dilated[:-i] |= is_spike[i:]
            
        return ~is_spike_dilated

    def _get_segment_slices(self, mask):
        padded = np.concatenate(([False], mask, [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        slices = []
        for start, end in zip(transitions[0::2], transitions[1::2]):
            slices.append(slice(start, end))
        return slices

    def _plot_segmentation_results(self, raw_ts, r_pos, r_vel, r_acc, processed_data, valid_segments, mask):
        """Plots Raw (Blue) vs Filtered (Green) ON THE SAME TIMESTEPS."""
        import matplotlib.pyplot as plt
        raw_ts = raw_ts.flatten()

        signals = {
            "Positions": (r_pos, processed_data["positions"]),
            "Velocities": (r_vel, processed_data["velocities"]),
            "Accelerations": (r_acc, processed_data["accelerations"])
        }
        
        bad_segments = self._get_segment_slices(~mask)
        
        for name, (raw, clean_stacked) in signals.items():
            if raw.ndim == 1: raw = raw[:, np.newaxis]
            n_joints = raw.shape[1]
            fig, axes = plt.subplots(n_joints, 1, figsize=(10, 3 * n_joints), sharex=True)
            if n_joints == 1: axes = [axes]
            
            fig.suptitle(f'{name}: Segmentation Analysis', fontsize=16)

            for j in range(n_joints):
                ax = axes[j]
                
                # 1. Raw
                ax.plot(raw_ts, raw[:, j], label='Raw (Dirty)', color='blue', alpha=0.5, linewidth=1)
                
                # 2. Rejected
                for i_bad, seg in enumerate(bad_segments):
                    t_start = float(raw_ts[seg.start])
                    t_end = float(raw_ts[seg.stop - 1]) 
                    lbl = 'Rejected' if i_bad == 0 else ""
                    ax.axvspan(t_start, t_end, color='red', alpha=0.2, label=lbl)

                # 3. Processed (Segment by Segment on original time)
                current_idx = 0
                for i_seg, seg in enumerate(valid_segments):
                    seg_len = seg.stop - seg.start
                    clean_chunk = clean_stacked[current_idx : current_idx + seg_len, j]
                    time_chunk = raw_ts[seg]
                    
                    lbl = 'Processed' if i_seg == 0 else ""
                    ax.plot(time_chunk, clean_chunk, color='green', linewidth=2, linestyle='--', label=lbl)
                    current_idx += seg_len

                ax.set_ylabel(f'Joint {j}')
                if j == 0: ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.show()

    def _plot_single_signal(self, raw_data, valid_segments, name, raw, clean_stacked):
        """Helper for Torque plotting."""
        import matplotlib.pyplot as plt
        if raw.ndim == 1: raw = raw[:, np.newaxis]
        n_joints = raw.shape[1]
        
        raw_ts = np.array(raw_data["timestamps"]).flatten()
        
        mask = np.zeros(len(raw), dtype=bool)
        for seg in valid_segments:
            mask[seg] = True
        bad_segments = self._get_segment_slices(~mask)

        fig, axes = plt.subplots(n_joints, 1, figsize=(10, 3 * n_joints), sharex=True)
        if n_joints == 1: axes = [axes]
        
        fig.suptitle(f'{name.capitalize()}: Segmentation Analysis', fontsize=16)

        for j in range(n_joints):
            ax = axes[j]
            ax.plot(raw_ts, raw[:, j], label='Raw', color='blue', alpha=0.5)
            
            for i_bad, seg in enumerate(bad_segments):
                t_start = float(raw_ts[seg.start])
                t_end = float(raw_ts[seg.stop - 1])
                ax.axvspan(t_start, t_end, color='red', alpha=0.2)
                
            current_idx = 0
            for i_seg, seg in enumerate(valid_segments):
                seg_len = seg.stop - seg.start
                clean_chunk = clean_stacked[current_idx : current_idx + seg_len, j]
                time_chunk = raw_ts[seg]
                ax.plot(time_chunk, clean_chunk, color='green', linewidth=2, linestyle='--')
                current_idx += seg_len
            
            ax.set_ylabel(f'Joint {j}')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    def _eliminate_zero_columns(self, dynamic_regressor, standard_parameter, zero_tolerance):
        idx_eliminated, active_parameters = get_index_eliminate(dynamic_regressor, standard_parameter, tol_e=zero_tolerance)
        regressor_reduced = build_regressor_reduced(dynamic_regressor, idx_eliminated)
        return regressor_reduced, active_parameters, idx_eliminated
    
    def _apply_decimation(self, processed_data, num_samples, regressor_reduced, decimation_factor):
        from scipy import signal

        # Decimate torque data
        tau_decimated_list = []
        num_joints = len(self.identif_config["act_idxv"])

        for i in range(num_joints):
            tau_joint = processed_data["torques"][:, i]
            tau_dec = signal.decimate(tau_joint, q=decimation_factor,
                                      zero_phase=True)
            tau_decimated_list.append(tau_dec)

        # Concatenate decimated torque data
        tau_decimated = tau_decimated_list[0]
        for i in range(1, len(tau_decimated_list)):
            tau_decimated = np.append(tau_decimated, tau_decimated_list[i])

        # Decimate regressor matrix
        regressor_decimated = self._decimate_regressor_matrix(
            num_samples, regressor_reduced, decimation_factor)

        # Validate that decimated data is properly aligned
        if tau_decimated.shape[0] != regressor_decimated.shape[0]:
            raise ValueError(
                f"Decimated data size mismatch: "
                f"tau_decimated has {tau_decimated.shape[0]} samples, "
                f"regressor_decimated has {regressor_decimated.shape[0]} rows"
            )

        tau_decimated = tau_decimated
        regressor_decimated = regressor_decimated
        return tau_decimated, regressor_decimated

    def _decimate_regressor_matrix(self, num_samples, regressor_reduced, decimation_factor):
        from scipy import signal

        num_joints = len(self.identif_config["act_idxv"])
        regressor_list = []

        for i in range(num_joints):
            # Extract rows corresponding to joint i
            start_idx = self.identif_config["act_idxv"][i] * num_samples
            end_idx = (self.identif_config["act_idxv"][i] + 1) * num_samples

            joint_regressor_decimated = []
            for j in range(regressor_reduced.shape[1]):
                column_data = regressor_reduced[start_idx:end_idx, j]
                decimated_column = signal.decimate(
                    column_data, q=decimation_factor, zero_phase=True)
                joint_regressor_decimated.append(decimated_column)

            # Reconstruct matrix for this joint
            joint_matrix = np.zeros((len(joint_regressor_decimated[0]),
                                     len(joint_regressor_decimated)))
            for k, column in enumerate(joint_regressor_decimated):
                joint_matrix[:, k] = column
            regressor_list.append(joint_matrix)

        # Concatenate all joint matrices
        total_rows = sum(matrix.shape[0] for matrix in regressor_list)
        regressor_decimated = np.zeros((total_rows,
                                        regressor_list[0].shape[1]))

        current_row = 0
        for matrix in regressor_list:
            next_row = current_row + matrix.shape[0]
            regressor_decimated[current_row:next_row, :] = matrix
            current_row = next_row

        return regressor_decimated

    def _calculate_base_parameters(self, standard_parameter, tau_processed, regressor_processed,
                                   active_parameters):
        from figaroh.tools.qrdecomposition import double_QR

        # Perform QR decomposition
        W_base, base_param_dict, base_parameters, phi_base, phi_std = \
            double_QR(tau_processed, regressor_processed, active_parameters,
                      standard_parameter)

        # Calculate torque estimation (avoid redundant computation)
        tau_estimated = np.dot(W_base, phi_base)

        return {
            "base_regressor": W_base,
            "base_param_dict": base_param_dict,
            "base_parameters": base_parameters,
            "phi_base": phi_base,
            "tau_estimated": tau_estimated,
            "tau_processed": tau_processed,
        }

    def get_link_geometric_bounds(self):
        """
        Extracts the bounding box of collision meshes for each active joint.
        Returns a dict: {joint_name: [[min_x, y, z], [max_x, y, z]]}
        """
        bounds_dict = {}
        geom_model = self.robot.collision_model
        
        for j_name in self.identif_config["active_joints"]:
            joint_id = self.model.getJointId(j_name)
            
            # Find all collision objects attached to this joint
            joint_vertices = []
            for geom_obj in geom_model.geometryObjects:
                if geom_obj.parentJoint == joint_id:
                    # Get mesh and apply its placement relative to joint
                    mesh = trimesh.load(geom_obj.meshPath)
                    T = geom_obj.placement.homogeneous
                    mesh.apply_transform(T)
                    joint_vertices.append(mesh.vertices)
            
            if joint_vertices:
                all_v = np.vstack(joint_vertices)
                # Calculate AABB [min_coords, max_coords]
                bounds_dict[j_name] = [np.min(all_v, axis=0), np.max(all_v, axis=0)]
            else:
                # Fallback if no mesh found (e.g., 5cm box)
                bounds_dict[j_name] = [np.array([-0.05]*3), np.array([0.05]*3)]
                
        return bounds_dict
        
    def solve_global(self, zero_tolerance=1e-6, plotting=True, save_results=False):
        print(f"\n[Global ID] Starting WTLS Identification (Reference Function)...")
        
        nb_joints = len(self.identif_config["active_joints"])
        
        q_rand = np.random.uniform(low=-6, high=6, size=(self.num_samples_A, self.model.nq))
        dq_rand = np.random.uniform(low=-6, high=6, size=(self.num_samples_A, self.model.nv))
        ddq_rand = np.random.uniform(low=-6, high=6, size=(self.num_samples_A, self.model.nv))
        
        W_rand = build_regressor_basic(self.robot, q_rand, dq_rand, ddq_rand, self.identif_config)
        params_standard = get_standard_parameters(self.model, self.identif_config)
                    
        idx_e, params_r = get_index_eliminate(W_rand, params_standard, tol_e=zero_tolerance)
        W_e = build_regressor_reduced(W_rand, idx_e)
        _, params_base, idx_base = get_baseParams(W_e, params_r, params_standard)
        
        # --- 2. Experiment A (Loaded) ---
        W_A = build_regressor_basic(
            self.robot, 
            self.processed_data_A["positions"], 
            self.processed_data_A["velocities"], 
            self.processed_data_A["accelerations"], 
            self.identif_config
        )
        W_e_A = build_regressor_reduced(W_A, idx_e)
        W_base_A = W_e_A[:, idx_base]
        cond_num = np.linalg.cond(W_base_A)
        print(f"Condition Number = {cond_num:.2e}")
        
        # --- 3. Experiment B (Loaded) ---
        W_B = build_regressor_basic(
            self.robot, 
            self.processed_data_B["positions"], 
            self.processed_data_B["velocities"], 
            self.processed_data_B["accelerations"], 
            self.identif_config
        )
        
        # --- 4. Experiment Test (Unloaded) ---
        W_test = build_regressor_basic(
            self.robot, 
            self.processed_data_test["positions"], 
            self.processed_data_test["velocities"], 
            self.processed_data_test["accelerations"], 
            self.identif_config
            )
        
        # --- Get mesh bounds ---
        mesh_bounds = self.get_link_geometric_bounds()
        
        # # --- 3. PARETO FRONT ANALYSIS ---         
        # lambda_r_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 5_000.0, 10_000.0]
        # errors = []
        # deviations = []
        
        # print(f"\n{'Lambda':<10} | {'RMSE (Nm)':<12} | {'Dev from CAD':<12}")
        # print("-" * 40)

        # for l in lambda_r_values:
        #     # 1. Solve
        #     res = solve_differential_LMI_OLS(
        #         W_A, self.processed_data_A["torques"], self.identif_config["mass_load_A"],
        #         W_B, self.processed_data_B["torques"], self.identif_config["mass_load_B"],
        #         load_joint_idx=self.identif_config["which_body_loaded"],
        #         nb_joints=nb_joints,
        #         joint_names=self.identif_config["active_joints"],
        #         phi_cad_dict=params_standard,
        #         armature_vals=self.identif_config["armatures"],
        #         mesh_bounds=mesh_bounds,
        #         lambda_r=l,              
        #         lambda_l=l,
        #     )
            
        #     # Unpack results
        #     phi_rob, phi_lA, phi_lB, k_tau, (tau_load_A_phys, tau_load_B_phys), (term_fit_norm, term_rob, term_load) = res
            
        #     # 2. COMPUTE PHYSICAL RMSE (Manual Reconstruction)
        #     # Reconstruct Robot Torque
        #     tau_identified_A_scaled = (W_A @ phi_rob + tau_load_A_phys).reshape((self.num_samples_A, nb_joints), order='F')
        #     tau_identified_B_scaled = (W_B @ phi_rob + tau_load_B_phys).reshape((self.num_samples_B, nb_joints), order='F')
            
        #     # Scaled Measured Torque (Y * k)
        #     tau_meas_A_scaled = self.processed_data_A["torques"] * k_tau
        #     tau_meas_B_scaled = self.processed_data_B["torques"] * k_tau
            
        #     # Compute Global RMSE (Concatenate errors from A and B)
        #     err_A = (tau_identified_A_scaled - tau_meas_A_scaled).flatten()
        #     err_B = (tau_identified_B_scaled - tau_meas_B_scaled).flatten()
        #     all_errors = np.concatenate([err_A, err_B])
            
        #     rmse_phys = np.sqrt(np.mean(all_errors**2))
        #     errors.append(rmse_phys)
            
        #     # Compute Deviation
        #     if l > 1e-9:
        #         dev_norm = np.sqrt(term_rob / l)
        #     else:
        #         dev_norm = 0.0 
        #     deviations.append(dev_norm)
            
        #     print(f"{l:<10.1f} | {rmse_phys:<12.4f} | {dev_norm:<12.4f}")

        # # --- 4. Plotting the L-Curve ---
        # if plotting:
        #     import matplotlib.pyplot as plt
        #     fig, ax1 = plt.subplots(figsize=(8, 5))

        #     color = 'tab:red'
        #     ax1.set_xlabel('Lambda R (Log Scale)')
        #     ax1.set_ylabel('Physical RMSE (Nm)', color=color) # Label updated
        #     ax1.plot(lambda_r_values, errors, color=color, marker='o', label='Fit Error')
        #     ax1.tick_params(axis='y', labelcolor=color)
        #     ax1.set_xscale('log')
        #     ax1.grid(True, which="both", ls="-", alpha=0.3)

        #     ax2 = ax1.twinx()  
        #     color = 'tab:blue'
        #     ax2.set_ylabel('Parameter Deviation (Norm)', color=color)
        #     ax2.plot(lambda_r_values, deviations, color=color, marker='x', linestyle='--', label='CAD Deviation')
        #     ax2.tick_params(axis='y', labelcolor=color)

        #     plt.title('L-Curve: Accuracy (Nm) vs Fidelity')
        #     plt.tight_layout()
        #     plt.show()
        
        phi_robot_val, phi_load_A_val, phi_load_B_val, k_tau_val, (tau_load_A, tau_load_B), (term_fit_val, term_robot_val, term_load_val) = solve_differential_LMI_OLS(
                W_A, self.processed_data_A["torques"], self.identif_config["mass_load_A"],
                W_B, self.processed_data_B["torques"], self.identif_config["mass_load_B"],
                load_joint_idx=self.identif_config["which_body_loaded"],
                nb_joints=nb_joints,
                joint_names=self.identif_config["active_joints"],
                phi_cad_dict=params_standard,
                armature_vals=self.identif_config["armatures"],
                mesh_bounds=mesh_bounds,
                lambda_r=0.0001,              
                lambda_l=0.1,           
            )
        
        print(phi_robot_val-list(params_standard.values()))
        print(phi_load_A_val, phi_load_B_val)
                
        # --- 6. PLOTTING VALIDATION (CORRECTED) ---
        if plotting:
            print(f"Gains: {k_tau_val}")
            
            # === PLOT 1: EXPERIMENT A (ROBOT ONLY) ===
            def plot_validation(title, tau_model, tau_corrected, k_tau_val, tau_nominal=None):
                import matplotlib.pyplot as plt
                    
                figA, axsA = plt.subplots(nb_joints, 1, figsize=(10, 3 * nb_joints), sharex=True)
                figA.suptitle(title, fontsize=16)
                for j in range(nb_joints):
                    joint_name = self.identif_config["active_joints"][j] if "active_joints" in self.identif_config else f"Joint {j}"
                    ax = axsA[j]
                    if tau_nominal is not None:
                        ax.plot(tau_nominal[:, j], label='Nominal (x Gain)', color='red')
                    ax.plot(tau_corrected[:, j], label='Measured (x Gain)', color='blue')
                    ax.plot(tau_model[:, j], label='Identified Model', color='green')
                    ax.set_title(f"Joint {j}: Gain = {k_tau_val[j]:.4f}")
                    ax.set_ylabel(f'{joint_name}\n(Nm)')
                    ax.grid(True, alpha=0.3)
                    if j == 0: ax.legend(loc='upper right')
                
                axsA[-1].set_xlabel('Samples')
                plt.tight_layout()
                plt.show()
                
                if tau_nominal is not None:
                    figA, axsA = plt.subplots(nb_joints, 1, figsize=(10, 3 * nb_joints), sharex=True)
                    figA.suptitle(title, fontsize=16)
                    for j in range(nb_joints):
                        joint_name = self.identif_config["active_joints"][j] if "active_joints" in self.identif_config else f"Joint {j}"
                        ax = axsA[j]
                        ax.plot(np.abs(tau_corrected[:, j]-tau_nominal[:, j]), label='abs(Measured (x Gain) - Nominal (x Gain))', color='red')
                        ax.plot(np.abs(tau_corrected[:, j]-tau_model[:, j]), label='abs(Measured (x Gain) - Identified Model)', color='green')
                        ax.set_title(f"Joint {j}: Gain = {k_tau_val[j]:.4f}")
                        ax.set_ylabel(f'{joint_name}\n(Nm)')
                        ax.grid(True, alpha=0.3)
                        if j == 0: ax.legend(loc='upper right')
                    
                    axsA[-1].set_xlabel('Samples')
                    plt.tight_layout()
                    plt.show()

            tau_corrected_A = self.processed_data_A["torques"] * k_tau_val
            tau_model_A_flat = W_A @ phi_robot_val + tau_load_A
            tau_model_A = tau_model_A_flat.reshape((self.num_samples_A, nb_joints), order='F')
            tau_nominal_A_flat = W_A @ list(params_standard.values()) + tau_load_A
            tau_nominal_A = tau_nominal_A_flat.reshape((self.num_samples_A, nb_joints), order='F')
            plot_validation("LOADED (LEIGHT)", tau_model_A, tau_corrected_A, k_tau_val, tau_nominal=tau_nominal_A)
            
            tau_corrected_B = self.processed_data_B["torques"] * k_tau_val
            tau_model_B_flat = W_B @ phi_robot_val + tau_load_B
            tau_model_B = tau_model_B_flat.reshape((self.num_samples_B, nb_joints), order='F')
            tau_nominal_B_flat = W_B @ list(params_standard.values()) + tau_load_B
            tau_nominal_B = tau_nominal_B_flat.reshape((self.num_samples_B, nb_joints), order='F')
            plot_validation("LOADED (HEAVY)", tau_model_B, tau_corrected_B, k_tau_val, tau_nominal=tau_nominal_B)
            
            tau_corrected_test = self.processed_data_test["torques"] * k_tau_val
            tau_model_test_flat = W_test @ phi_robot_val
            tau_model_test = tau_model_test_flat.reshape((self.num_samples_test, nb_joints), order='F')
            tau_nominal_test_flat = W_test @ list(params_standard.values())
            tau_nominal_test = tau_nominal_test_flat.reshape((self.num_samples_test, nb_joints), order='F')
            
            plot_validation("TEST", tau_model_test, tau_corrected_test, k_tau_val, tau_nominal=tau_nominal_test)
        
        return phi_robot_val