"""
Custom tools and utilities specific to ROBOT_TITLE robot.

This module contains helper functions, custom classes, and utilities
that are specific to working with the ROBOT_TITLE robot.
"""

import pandas as pd
import re
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pinocchio as pin
from os.path import abspath
from figaroh.tools.robot import Robot
from figaroh.identification.base_identification import BaseIdentification
from figaroh.tools.solver import LinearSolver


def load_robot_model(robot_name: str = "robot_lower", **kwargs):
    """
    Load the ROBOT_TITLE robot model.
    
    Args:
        robot_name: Name/identifier for the robot
        **kwargs: Additional arguments for robot loading
        
    Returns:
        Robot: Loaded robot model
    """
    from figaroh.tools.robot import load_robot
    
    # TODO: Update with correct parameters for your robot
    robot = load_robot(
        robot_name=robot_name,
        load_by_urdf=True,
        robot_pkg="robot_lower_description",
        **kwargs
    )
    
    return robot

class H1v2Identification(BaseIdentification):
    """H1v2-specific dynamic parameter identification class."""
    
    P_BOUND = (0, None)    # Positive or zero (e.g., Mass, Friction, Actuator Inertia)
    N_BOUND = (None, None) # No bounds (e.g., Offsets, Center-of-Mass products, Cross-Inertia)
    
    def __init__(self, robot, config_file="config/h1v2_config.yaml"):
        """Initialize H1v2 identification with robot model and configuration.
        
        Args:
            robot: H1v2 robot model loaded with FIGAROH
            config_file: Path to H1v2 configuration YAML file
        """
        super().__init__(robot, config_file)
        print("H1v2Identification initialized for H1v2 robot")
    
    def load_trajectory_data(self):
        """Load and process CSV data for TIAGo robot."""
        ts = pd.read_csv(
            abspath(self.identif_config["pos_data"]), usecols=[0]
        ).to_numpy()
        pos = pd.read_csv(abspath(self.identif_config["pos_data"]))
        vel = pd.read_csv(abspath(self.identif_config["vel_data"]))
        eff = pd.read_csv(abspath(self.identif_config["torque_data"]))

        cols = {"pos": [], "vel": [], "eff": []}
        for jn in self.identif_config["active_joints"]:
            cols["pos"].extend([col for col in pos.columns if jn in col])
            cols["vel"].extend([col for col in vel.columns if jn in col])
            cols["eff"].extend([col for col in eff.columns if jn in col])

        q = pos[cols["pos"]].to_numpy()
        dq = vel[cols["vel"]].to_numpy()
        tau = eff[cols["eff"]].to_numpy()
        self.raw_data = {
            "timestamps": ts,
            "positions": q,
            "velocities": dq,
            "accelerations": None,
            "torques": tau
        }
        return self.raw_data

    def _apply_filters(self, data, delta=0.001, window_length=11, polyorder=11, deriv=0, **kwargs):
        """
        Applies a Savitzky-Golay filter with derivative support.
        
        Args:
            data (np.array): Input data.
            delta (float): The time step (sampling spacing). Essential for correct derivative scaling.
            window_length (int): Length of the filter window.
            polyorder (int): Order of the polynomial.
            deriv (int): The order of the derivative to compute (0=smooth, 1=vel, 2=acc).
        """
        if len(data) == 0:
            return data

        # SavGol requires window_length to be odd
        if window_length % 2 == 0:
            window_length += 1
            
        # SavGol requires window_length <= size of data
        if window_length > len(data):
            window_length = len(data) if len(data) % 2 != 0 else len(data) - 1
            if window_length < polyorder + 2:
                # Fallback if data is too short for the requested polyorder
                return data 

        # axis=0 ensures we filter over time (rows), not across joints (cols)
        # delta ensures derivatives are scaled correctly by time
        return savgol_filter(data, window_length=window_length, polyorder=polyorder, 
                             deriv=deriv, delta=delta, axis=0, **kwargs)
    
    def filter_kinematics_data(self, filter_config=None):
        self.latest_filter_config = filter_config

        # --- 1. VALIDATION ---
        if self.raw_data.get("timestamps") is None:
            raise ValueError("Timestamps are required for data processing")
        if self.raw_data.get("positions") is None:
            raise ValueError("Position data is required for processing")
        if self.raw_data.get("velocities") is None:
            raise ValueError("Velocity data is required for processing")

        raw_ts = np.array(self.raw_data["timestamps"]).flatten() 
        raw_pos = np.array(self.raw_data["positions"])
        raw_vel = np.array(self.raw_data["velocities"])
        
        dts = np.diff(raw_ts)
        median_dt = np.median(dts) if len(dts) > 0 else 0.001
        
        # --- 2. DETECTION PASS ---
        dirty_acc = self._differentiate_signal(raw_ts, raw_vel, method=filter_config['differentiation_method'])
        valid_mask = self._get_valid_mask_from_accel(dirty_acc, sensitivity=50.0, dilation=50)
        self.valid_segments = self._get_segment_slices(valid_mask)
        
        # 3a. Smooth Position (deriv=0)
        filtered_pos = self._apply_filters(raw_pos, delta=median_dt, deriv=0, **filter_config['filter_params'])
        filtered_vel = self._apply_filters(raw_pos, delta=median_dt, deriv=1, **filter_config['filter_params'])
        filtered_acc = self._apply_filters(raw_pos, delta=median_dt, deriv=2, **filter_config['filter_params'])
        
        chunks = {"positions": [], "velocities": [], "accelerations": []}     

        for seg in self.valid_segments:
            chunks["positions"].append(filtered_pos[seg])
            chunks["velocities"].append(filtered_vel[seg])
            chunks["accelerations"].append(filtered_acc[seg])

        # --- 4. CONCATENATION ---
        self.processed_data = {}
        if chunks["positions"]:
            self.processed_data["positions"] = np.vstack(chunks["positions"])
            self.processed_data["velocities"] = np.vstack(chunks["velocities"])
            self.processed_data["accelerations"] = np.vstack(chunks["accelerations"])

            total_samples = len(self.processed_data["positions"])
            self.processed_data["timestamps"] = np.arange(total_samples) * median_dt + raw_ts[0]
        else:
            print("Warning: All data segments were rejected.")
            self.processed_data = {k: np.array([]) for k in ["positions", "velocities", "accelerations", "timestamps"]}

        # --- 5. PLOTTING ---
        # Note: We plot raw velocities/accels (calculated roughly) vs SavGol results
        raw_vel_plot = np.array(self.raw_data["velocities"]) if self.raw_data.get("velocities") is not None else raw_vel
        self._plot_segmentation_results(raw_ts, raw_pos, raw_vel_plot, dirty_acc, valid_mask)

    def process_torque_data(self):
        """Process torque data using the same segments and filters."""
        if not hasattr(self, 'valid_segments') or self.valid_segments is None:
            raise ValueError("Run filter_kinematics_data first.")
            
        if self.raw_data.get("torques") is None:
            print("No torque data found.")
            return None

        raw_torques = np.array(self.raw_data["torques"])
        # Get delta for consistency
        raw_ts = np.array(self.raw_data["timestamps"]).flatten()
        dts = np.diff(raw_ts)
        median_dt = np.median(dts) if len(dts) > 0 else 0.001
        raw_torques = self._apply_filters(raw_torques, delta=median_dt, deriv=0, **self.latest_filter_config['filter_params'])

        torque_chunks = []

        for seg in self.valid_segments:
            seg_tau = raw_torques[seg]
            torque_chunks.append(seg_tau)

        if torque_chunks:
            self.processed_data["torques"] = np.vstack(torque_chunks)
            self._plot_single_signal("torques", raw_torques, self.processed_data["torques"])
            return self.processed_data["torques"]
        return np.array([])

    # --- HELPER METHODS ---
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

    def _plot_segmentation_results(self, raw_ts, r_pos, r_vel, r_acc, mask):
        """Plots Raw (Blue) vs Filtered (Green) ON THE SAME TIMESTEPS."""
        import matplotlib.pyplot as plt
        raw_ts = raw_ts.flatten()

        signals = {
            "Positions": (r_pos, self.processed_data["positions"]),
            "Velocities": (r_vel, self.processed_data["velocities"]),
            "Accelerations": (r_acc, self.processed_data["accelerations"])
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
                for i_seg, seg in enumerate(self.valid_segments):
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

    def _plot_single_signal(self, name, raw, clean_stacked):
        """Helper for Torque plotting."""
        import matplotlib.pyplot as plt
        if raw.ndim == 1: raw = raw[:, np.newaxis]
        n_joints = raw.shape[1]
        
        raw_ts = np.array(self.raw_data["timestamps"]).flatten()
        
        mask = np.zeros(len(raw), dtype=bool)
        for seg in self.valid_segments:
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
            for i_seg, seg in enumerate(self.valid_segments):
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
    
    def plot_results(self):
        """
        Plots measured vs identified vs nominal torques and their respective errors.
        """
        try:
            raw_torques = self.raw_data["torques"]
            n_joints = raw_torques.shape[1]
        except (KeyError, AttributeError, IndexError):
            n_joints = 7
            print(f"Warning: Could not determine n_joints from raw_data. Defaulting to {n_joints}.")

        tau_measured = self.result.get("torque processed", np.array([]))
        tau_identified = self.result.get("torque estimated", np.array([]))
        
        if len(tau_measured) == 0 or len(tau_identified) == 0:
            print("No torque data available for plotting")
            return
        
        # --- Reconstruct nominal torques ---
        # We calculate this first to ensure dimensions match before reshaping
        tau_nominal = np.zeros((self.processed_data["positions"].shape))
        for i in range(tau_nominal.shape[0]):
            tau_nominal[i] = pin.rnea(self.model, 
                                            self.data, 
                                            self.processed_data["positions"][i], 
                                            self.processed_data["velocities"][i], 
                                            self.processed_data["accelerations"][i]
                                        )
        tau_nominal = tau_nominal[:, self.identif_config["act_idxq"]]
        
        # --- Reshape Data ---
        if tau_measured.ndim == 1:
            tau_measured = tau_measured.reshape(-1, n_joints, order='F')
        if tau_identified.ndim == 1:
            tau_identified = tau_identified.reshape(-1, n_joints, order='F')

        # Ensure shapes match across all three signals
        n_samples = min(len(tau_measured), len(tau_identified), len(tau_nominal))
        tau_measured = tau_measured[:n_samples]
        tau_identified = tau_identified[:n_samples]
        tau_nominal = tau_nominal[:n_samples]
        
        # --- Setup Plot ---
        fig, axes = plt.subplots(n_joints, 2, figsize=(14, 3 * n_joints), sharex='col')
        
        if n_joints == 1:
            axes = axes.reshape(1, -1)

        time_axis = np.arange(n_samples)

        # --- Loop over joints ---
        for j in range(n_joints):
            # 1. Calculate Errors
            error_ident = tau_measured[:, j] - tau_identified[:, j]
            error_nom = tau_measured[:, j] - tau_nominal[:, j]

            # 2. Calculate RMSEs
            rmse_ident = np.sqrt(np.mean(error_ident**2))
            rmse_nom = np.sqrt(np.mean(error_nom**2))

            # --- Left Column: Signal Comparison ---
            ax_comp = axes[j, 0]
            ax_comp.plot(time_axis, tau_measured[:, j], label="Measured", color='#1f77b4', alpha=0.6)
            ax_comp.plot(time_axis, tau_identified[:, j], label="Identified", color='#ff7f0e', alpha=0.8, linestyle='--')
            ax_comp.plot(time_axis, tau_nominal[:, j], label="Nominal", color="#04d832", alpha=0.5, linestyle='--')
            
            ax_comp.set_ylabel(f'Joint {j+1}\nTorque (Nm)', fontweight='bold')
            
            # Enhanced Text Box with both RMSEs
            stats_text = (f"RMSE Ident: {rmse_ident:.4f}\n"
                        f"RMSE Nom:   {rmse_nom:.4f}")
            ax_comp.text(0.02, 0.95, stats_text, transform=ax_comp.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'), 
                            verticalalignment='top', fontfamily='monospace', fontsize=9)
            
            ax_comp.grid(True, alpha=0.3)
            
            if j == 0:
                ax_comp.set_title("Torque Tracking (Measured vs Models)")
                ax_comp.legend(loc='upper right', framealpha=0.9)

            # --- Right Column: Error Comparison ---
            ax_err = axes[j, 1]
            
            # Plot Nominal Error first (background, typically larger)
            ax_err.plot(time_axis, error_nom, label="Meas - Nom", color='#04d832', alpha=0.4)
            # Plot Identified Error second (foreground, typically smaller)
            ax_err.plot(time_axis, error_ident, label="Meas - Ident", color='#d62728', alpha=0.8)
            
            ax_err.axhline(0, color='k', linewidth=0.8, alpha=0.5) 
            ax_err.set_ylabel('Error (Nm)')
            ax_err.grid(True, alpha=0.3)
            
            if j == 0:
                ax_err.set_title("Residual Errors")
                ax_err.legend(loc='upper right')

            # X-labels only on bottom
            if j == n_joints - 1:
                ax_comp.set_xlabel('Sample')
                ax_err.set_xlabel('Sample')

        plt.suptitle(f'{self.__class__.__name__} Results Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
        plt.show()
        
    def solve_with_custom_solver(
        self, method='lstsq', regularization=None, alpha=0.0,
        constraints=None, decimate=False,
        decimation_factor=1, zero_tolerance=0.001,
        plotting=False, save_results=False, **solver_kwargs
    ):
        """
        Alternative solving method using advanced linear solver.

        This method provides more flexibility than the default QR-based
        solve(), offering multiple solving methods, regularization, and
        constraints.

        Args:
            method (str): Solving method ('lstsq', 'ridge', 'lasso',
                'constrained', etc.)
            regularization (str): Regularization type ('l1', 'l2',
                'elastic_net')
            alpha (float): Regularization strength
            constraints (dict): Linear constraints
            decimate (bool): Whether to apply decimation
            decimation_factor (int): Decimation factor if decimate=True
            zero_tolerance (float): Tolerance for eliminating zero columns
            plotting (bool): Whether to generate plots
            save_results (bool): Whether to save parameters to file
            **solver_kwargs: Additional arguments for LinearSolver

        Returns:
            ndarray: Identified base parameters
        """
        print(f"Starting {self.__class__.__name__} identification "
              f"with custom solver...")

        # Validate prerequisites
        self._validate_prerequisites()

        # Step 1: Eliminate zero columns
        regressor_reduced, active_params = self._eliminate_zero_columns(
            zero_tolerance
        )

        # Step 2: Apply decimation if requested
        if decimate:
            tau_processed, W_processed = self._apply_decimation(
                regressor_reduced, decimation_factor
            )
        else:
            tau_processed, W_processed = self._prepare_undecimated_data(
                regressor_reduced
            )
        
        from figaroh.tools.qrdecomposition import double_QR

        W_base, _, base_parameters, _, phi_std = \
            double_QR(
                tau_processed, W_processed, active_params,
                self.standard_parameter
            )

        # Step 3: Solve using custom solver
        solver = LinearSolver(
            method=method,
            regularization=regularization,
            alpha=alpha,
            constraints=constraints,
            bounds=self._get_bounds(base_parameters),
            verbose=True,
            **solver_kwargs
        )

        # Step 4: Compute base parameters using QR decomposition
        phi_base = solver.solve(W_base, tau_processed)
        base_param_dict = {param: phi_base[i] for i, param in enumerate(base_parameters)}
        
        # Store results
        self.dynamic_regressor_base = W_base
        self.phi_base = phi_base
        self.params_base = list(base_param_dict.keys())
        self.tau_identif = W_base @ phi_base
        self.tau_noised = tau_processed

        # Step 5: Compute quality metrics and store
        self._compute_quality_metrics()

        results = {
            "base_regressor": W_base,
            "base_param_dict": base_param_dict,
            "base_parameters": base_parameters,
            "phi_base": phi_base,
            "tau_estimated": self.tau_identif,
            "tau_processed": tau_processed,
            "solver_info": solver.solver_info,
            "solver_method": method,
            "regularization": regularization,
            "alpha": alpha
        }

        self._store_results(results)

        # Step 6: Optional plotting
        if plotting:
            self.plot_results()

        # Step 7: Optional parameter saving
        if save_results:
            self.save_results()

        print(f"  RMSE: {self.rms_error:.6f}")
        print(f"  Correlation: {self.correlation:.6f}")

        return self.phi_base
    
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