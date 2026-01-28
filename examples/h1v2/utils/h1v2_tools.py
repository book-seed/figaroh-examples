"""
Custom tools and utilities specific to ROBOT_TITLE robot.

This module contains helper functions, custom classes, and utilities
that are specific to working with the ROBOT_TITLE robot.
"""

import pandas as pd
import re
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
    build_total_regressor_current,
    solve_LMI_OLS, 
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
    
    def initialize_global(self, data_pathes_A, data_pathes_B, truncate_A=None, truncate_B=None):
        # Experiment A
        self.processed_data_A, self.num_samples_A = self.process_data(data_pathes=data_pathes_A, truncate=truncate_A)
        
        # Experiment B
        self.processed_data_B, self.num_samples_B = self.process_data(data_pathes=data_pathes_B, truncate=truncate_B)
        
    def process_data(self, data_pathes, truncate=None):
        """Load and process data"""
        filter_config = self.filter_config
        raw_data = self.load_trajectory_data(data_pathes)
        raw_data = self._truncate_data(raw_data, truncate)
        
        # 1. Filter Kinematics (Butterworth)
        processed_data, valid_segments = self.filter_kinematics_data(raw_data, filter_config)
        
        # 2. Filter Torques (Butterworth) using same segments
        processed_data["torques"] = self.process_torque_data(raw_data, processed_data, valid_segments)
        # processed_data = self._align_signals(processed_data, shift_amount=-3)
        
        # 4. Build Full Config
        num_samples = processed_data["positions"].shape[0]
        processed_data = self._build_full_configuration(processed_data, num_samples)
                
        return processed_data, num_samples

    def _align_signals(self, processed_data, shift_amount):
        """
        Shifts Torques relative to Kinematics to fix time lag.
        
        Args:
            processed_data: Dictionary with 'torques', 'positions', etc.
            shift_amount (int): 
                If Lag is POSITIVE (Torque is late), use NEGATIVE shift (e.g., -3).
                If Lag is NEGATIVE (Torque is early), use POSITIVE shift (e.g., +3).
        """
        if shift_amount == 0:
            return processed_data
            
        print(f"  -> Aligning Data: Rolling Torque by {shift_amount} samples")
        
        # 1. Roll the Torque Array
        # np.roll moves data. Data that "falls off" the end wraps to the beginning.
        processed_data["torques"] = np.roll(processed_data["torques"], shift_amount, axis=0)
        
        # 2. Slice off the "Wrapped" Garbage
        # We must delete the samples that wrapped around to prevent spikes.
        # We also trim the Kinematics to match the new length.
        if shift_amount < 0:
            # Shifted LEFT (Backwards). The last 'N' samples are now garbage (wrapped from start).
            # We also lose the first 'N' samples of validity.
            # Crop END of arrays.
            cut = abs(shift_amount)
            for key in ["positions", "velocities", "accelerations", "torques", "timestamps"]:
                if key in processed_data:
                    processed_data[key] = processed_data[key][:-cut]
                    
        elif shift_amount > 0:
            # Shifted RIGHT (Forwards). The first 'N' samples are garbage.
            # Crop START of arrays.
            cut = shift_amount
            for key in ["positions", "velocities", "accelerations", "torques", "timestamps"]:
                if key in processed_data:
                    processed_data[key] = processed_data[key][cut:]
                    
        return processed_data
    
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
        filtered_acc = self._apply_butterworth(raw_acc_calc, median_dt)
        
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
        
    def _get_bounds(self, variable_list: list[str]) -> list[tuple]:
        """
        Dynamically determines optimization bounds based on the Leading Term 
        and the algebraic structure of the Base Parameter linear combination.
        
        P_BOUND (0, +inf): Principal Inertias (Ixx, Iyy, Izz, Ia), Friction (fv, fs), Mass (m).
        N_BOUND (-inf, +inf): Static Moments (mx, my, mz), Products of Inertia (Ixy...), 
                            Offsets, and DIFFERENCES of Principal Inertias (e.g. Ixx - Iyy).
        """
        unbounded_prefixes = ('mx', 'my', 'mz', 'off', 'Ixy', 'Ixz', 'Iyz')
    
        subtracted_inertia_pattern = re.compile(r'-\s*(?:[\d\.]+(?:e[+-]?\d+)?\*)?I[xyz]{2}')

        dynamic_bounds = []

        for var_string in variable_list:
            leading_term = var_string.split()[0]
            bound_type = self.P_BOUND
            
            # --- Check 1: Is the leading term inherently unbounded? ---
            if leading_term.startswith(unbounded_prefixes):
                bound_type = self.N_BOUND
            
            # --- Check 2: Special Case - Difference of Principal Inertias ---
                if subtracted_inertia_pattern.search(var_string):
                    bound_type = self.N_BOUND
                            
            dynamic_bounds.append(bound_type)

        return dynamic_bounds
        
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
        
        # --- 2. Experiment A (Unloaded) ---
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
        
        tau_A_armature = self.processed_data_A["accelerations"] * np.array(self.identif_config["armatures"])

        # --- 3. Experiment B (Loaded) ---
        W_B = build_regressor_basic(
            self.robot, 
            self.processed_data_B["positions"], 
            self.processed_data_B["velocities"], 
            self.processed_data_B["accelerations"], 
            self.identif_config
        )
        
        tau_B_armature = self.processed_data_B["accelerations"] * np.array(self.identif_config["armatures"])
        
        # # --- 3. GENERATE SYNTHETIC TORQUE ---
        # print("  -> Constructing Theoretical Parameter Vector...")
        # phi_standard_list = []

        # for i in range(1, self.model.njoints): 
        #     phi_i = self.model.inertias[i].toDynamicParameters()
        #     phi_standard_list.extend(phi_i)
            
        #     if self.identif_config.get("has_friction", False):
        #         phi_standard_list.extend([0.001, 0.1]) # fv=0.1, fs=0.1 (Fake Friction)
        #     if self.identif_config.get("has_actuator_inertia", False):
        #         phi_standard_list.append(0.01)       # ia (Fake Inertia)
        #     if self.identif_config.get("has_joint_offset", False):
        #         phi_standard_list.append(0.0)        # offset

        # phi_standard = np.array(phi_standard_list)
            
        # tau_A_synthetic = W_A @ phi_standard
        
        # m_load = self.identif_config["mass_load"]       
        # # r_in  = 0.03/2
        # r_in  = 0.0
        # r_out = 0.126/2
        # h_th  = 0.015
        # p_com = np.array([0.228 + 0.0775, 0.0, 0.0])
        
        # val_Ix = 0.5 * m_load * (r_in**2 + r_out**2)
        # val_Iy_Iz = (1.0/12.0) * m_load * (3*(r_in**2 + r_out**2) + h_th**2)    
        # I_load_at_com = np.diag([val_Ix, val_Iy_Iz, val_Iy_Iz])
        # p_norm_sq = np.dot(p_com, p_com)
        # p_outer   = np.outer(p_com, p_com)
        # I_load_at_origin = I_load_at_com + m_load * (p_norm_sq * np.eye(3) - p_outer)
        # h_load = m_load * p_com

        # delta_phi_load = np.array([
        #         m_load,                # Mass
        #         h_load[0], 
        #         h_load[1], 
        #         h_load[2],
        #         I_load_at_origin[0,0], # Ixx
        #         I_load_at_origin[0,1], # Ixy
        #         I_load_at_origin[1,1], # Iyy
        #         I_load_at_origin[0,2], # Ixz
        #         I_load_at_origin[1,2], # Iyz
        #         I_load_at_origin[2,2]  # Izz
        #     ])
        
        # cols_per_joint = 10 
        # if self.identif_config.get("has_friction", False): cols_per_joint += 2
        # if self.identif_config.get("has_actuator_inertia", False): cols_per_joint += 1
        # if self.identif_config.get("has_joint_offset", False): cols_per_joint += 1
        
        # idx_body = self.identif_config["which_body_loaded"] - 1
        # start_col = idx_body * cols_per_joint
        # W_payload_block = W_B[:, start_col : start_col + 10]
        # tau_payload = W_payload_block @ delta_phi_load
        
        # tau_B_synthetic = (W_B @ phi_standard) + tau_payload  
        
        # # gains = np.arange(1, nb_joints_partial + 1)
        # gains = np.ones(nb_joints_partial)
        # tau_A_flat = (tau_A_synthetic.reshape(nb_joints_partial, self.num_samples_A).T / gains).T.flatten() # + 0.3*np.random.rand(*tau_A_synthetic.shape)
        # tau_B_flat = (tau_B_synthetic.reshape(nb_joints_partial, self.num_samples_B).T / gains).T.flatten() # + 0.3*np.random.rand(*tau_B_synthetic.shape)
        
        # tau_A_flat = self.processed_data_A["torques"].flatten(order='F')  # comment to use pinocchio instead of mujoco for the torques and the TLS
        # tau_B_flat = self.processed_data_B["torques"].flatten(order='F')
        # tau_A_flat = filter_blocks(tau_A_flat, self.num_samples_A)
        # tau_B_flat = filter_blocks(tau_B_flat, self.num_samples_B)

        # # --- 5. Solve ---
        # W_tot, V_norm, residue = build_total_regressor_current(
        #     W_base_A, 
        #     W_base_B, 
        #     W_B, 
        #     tau_A_flat, 
        #     tau_B_flat, 
        #     self.identif_config,
        #     bounds
        # )
        
        # n_base_params = W_base_A.shape[1]
        # gains = V_norm[n_base_params : n_base_params + nb_joints_partial]
        
        phi_r, phi_l, k_vals, tau_load_B = solve_differential_LMI_OLS(
            W_A, self.processed_data_A["torques"], tau_A_armature,
            W_B, self.processed_data_B["torques"], tau_B_armature,
            mass_load_known=self.identif_config["mass_load"],  
            load_joint_idx=self.identif_config["which_body_loaded"],
            nb_joints=nb_joints,
            joint_names=self.identif_config["active_joints"],
            phi_cad_dict=params_standard,
            lambda_r=0.01,              
            lambda_l=0.1,              
        )
        print(phi_r)
        print(phi_l)
        print(phi_r.reshape(7, 13)-np.array(list(params_standard.values())).reshape(7, 13))
                
        # --- 6. PLOTTING VALIDATION (CORRECTED) ---
        if plotting:
            import matplotlib.pyplot as plt
            print(f"Gains: {k_vals}")
            
            # === PLOT 1: EXPERIMENT A (ROBOT ONLY) ===
            print("\n[Validation] Plotting Experiment A (Unloaded Robot Only)...")
            tau_corrected_A = self.processed_data_A["torques"] * k_vals
            tau_model_A_flat = W_A @ phi_r
            tau_model_A = tau_model_A_flat.reshape((self.num_samples_A, nb_joints), order='F') + tau_A_armature
            
            figA, axsA = plt.subplots(nb_joints, 1, figsize=(10, 3 * nb_joints), sharex=True)
            figA.suptitle(f'Validation A: UNLOADED (Base Params)', fontsize=16)
            for j in range(nb_joints):
                joint_name = self.identif_config["active_joints"][j] if "active_joints" in self.identif_config else f"Joint {j}"
                ax = axsA[j]
                ax.plot(tau_corrected_A[:, j], label='Measured (x Gain)', color='blue', alpha=0.6)
                ax.plot(tau_model_A[:, j], label='Identified Model', color='green', linestyle='--')
                ax.set_title(f"Joint {j}: Gain = {k_vals[j]:.4f}")
                ax.set_ylabel(f'{joint_name}\n(Nm)')
                ax.grid(True, alpha=0.3)
                if j == 0: ax.legend(loc='upper right')
            
            axsA[-1].set_xlabel('Samples')
            plt.tight_layout()
            plt.show()

            # === PLOT 2: EXPERIMENT B (LOADED) ===
            print("\n[Validation] Plotting Experiment B (Loaded)...")
            tau_corrected_B = self.processed_data_B["torques"] * k_vals
            tau_model_B_flat = W_B @ phi_r + tau_load_B
            tau_model_B = tau_model_B_flat.reshape((self.num_samples_B, nb_joints), order='F') + tau_B_armature

            figB, axsB = plt.subplots(nb_joints, 1, figsize=(10, 3 * nb_joints), sharex=True)
            figB.suptitle(f'Validation B: LOADED (Payload)', fontsize=16)

            for j in range(nb_joints):
                joint_name = self.identif_config["active_joints"][j] if "active_joints" in self.identif_config else f"Joint {j}"
                ax = axsB[j]
                ax.plot(tau_corrected_B[:, j], label='Measured (x Gain)', color='blue', alpha=0.6)
                ax.plot(tau_model_B[:, j], label='Identified Model', color='red', linestyle='--')
                ax.set_title(f"Joint {j}: Gain = {k_vals[j]:.4f}")
                ax.set_ylabel(f'{joint_name}\n(Nm)')
                ax.grid(True, alpha=0.3)
                if j == 0: ax.legend(loc='upper right')

            axsB[-1].set_xlabel('Samples')
            plt.tight_layout()
            plt.show()
            
        # # --- 7. OLS INDEPENDENT CHECK ---
        # print("\n[Sanity Check] Running Independent OLS for Exp A and B...")

        # # Solve A (Unloaded)
        # phi_ols_A = solve_LMI_OLS(W_A, self.processed_data_A["torques"].flatten(order='F'), nb_joints, list(params_standard.values()), 1.0)        
        # tau_ols_A_flat = W_A @ phi_ols_A

        # # Solve B (Loaded)
        # phi_ols_B = solve_LMI_OLS(W_B, self.processed_data_B["torques"].flatten(order='F'), nb_joints, list(params_standard.values()), 1.0)        
        # tau_ols_B_flat = W_B @ phi_ols_B
         
        # # --- 8. FANCY PLOTTING: Pinocchio vs Measured vs Identified ---
        # if plotting:
        #     import matplotlib.pyplot as plt
        #     import matplotlib.gridspec as gridspec
            
        #     # --- Helper to reshape flat vectors back to (Samples, Joints) ---
        #     def reshape_to_joints(flat_vec, n_samples, n_joints):
        #         return flat_vec.reshape((n_samples, n_joints), order='F')

        #     # Prepare Data for Plotting
        #     # A: Unloaded
        #     tau_meas_A = reshape_to_joints(tau_A_flat, self.num_samples_A, nb_joints_partial)
        #     tau_pin_A  = reshape_to_joints(tau_A_synthetic, self.num_samples_A, nb_joints_partial)
        #     tau_id_A   = reshape_to_joints(tau_ols_A_flat, self.num_samples_A, nb_joints_partial)
            
        #     # B: Loaded
        #     tau_meas_B = reshape_to_joints(tau_B_flat, self.num_samples_B, nb_joints_partial)
        #     tau_pin_B  = reshape_to_joints(tau_B_synthetic, self.num_samples_B, nb_joints_partial)
        #     tau_id_B   = reshape_to_joints(tau_ols_B_flat, self.num_samples_B, nb_joints_partial)

        #     def plot_fancy_comparison(title, tau_pin, tau_meas, tau_id, joint_names):
        #         n_j = len(joint_names)
        #         fig = plt.figure(figsize=(12, 3 * n_j))
        #         gs = gridspec.GridSpec(n_j, 1, figure=fig)
        #         fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
                
        #         colors = {'pin': '#2ca02c', 'meas': "#1f2eb4", 'id': '#d62728'} # Green, Blue, Red
                
        #         for j in range(n_j):
        #             ax = fig.add_subplot(gs[j, 0])
                    
        #             # Plot lines with slightly different widths/styles to see overlaps
        #             # 1. Measured (Thick, background)
        #             ax.plot(tau_meas[:, j], color=colors['meas'], linewidth=1.5, alpha=0.3, label='Measured')
                    
        #             # 2. Pinocchio (Medium, solid)
        #             ax.plot(tau_pin[:, j], color=colors['pin'], linewidth=1.5, linestyle='-', alpha=0.8, label='Nominal')
                    
        #             # 3. Identified OLS (Thin, dashed)
        #             ax.plot(tau_id[:, j], color=colors['id'], linewidth=1.5, linestyle='--', label='Identified')
                    
        #             # Styling
        #             ax.set_ylabel(f'{joint_names[j]}\nTorque (Nm)', fontsize=10, fontweight='bold')
        #             ax.grid(True, which='both', linestyle='--', alpha=0.4)
        #             ax.spines['top'].set_visible(False)
        #             ax.spines['right'].set_visible(False)
                    
        #             # Error text (RMSE between Pinocchio and ID)
        #             rmse = np.sqrt(np.mean((tau_pin[:, j] - tau_id[:, j])**2))
        #             ax.text(0.02, 0.90, f'Fit RMSE: {rmse:.4f} Nm', transform=ax.transAxes, 
        #                     fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        #             if j == 0:
        #                 ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False, fontsize=10)
                    
        #             if j == n_j - 1:
        #                 ax.set_xlabel('Time Samples', fontsize=11)
                
        #         plt.tight_layout()
        #         plt.subplots_adjust(top=0.92) # Make room for title/legend
        #         plt.show()

        #     # Active Joint Names
        #     j_names = [self.identif_config["active_joints"][i] for i in range(nb_joints_partial)]
                        
        #     print("\n[Plotting] 1. OLS Check - Experiment A (Unloaded)")
        #     plot_fancy_comparison("Sanity Check A: Unloaded Dynamics (OLS)", tau_pin_A, tau_meas_A, tau_id_A, j_names)
            
        #     print("\n[Plotting] 2. OLS Check - Experiment B (Loaded)")
        #     plot_fancy_comparison("Sanity Check B: Loaded Dynamics (OLS)", tau_pin_B, tau_meas_B, tau_id_B, j_names)