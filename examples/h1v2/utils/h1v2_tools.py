"""
Custom tools and utilities specific to ROBOT_TITLE robot.

This module contains helper functions, custom classes, and utilities
that are specific to working with the ROBOT_TITLE robot.
"""

import pandas as pd
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

    def process_torque_data(self):
        """Process torque data with TIAGo-specific motor constants."""
        import pinocchio as pin
        
        # Apply TIAGo-specific torque processing (reduction ratios, etc.)
        pin.computeSubtreeMasses(self.robot.model, self.robot.data)
        tau_processed = self.raw_data["torques"].copy()

        # for i, joint_name in enumerate(self.identif_config["active_joints"]):
        #     if joint_name == "torso_lift_joint":
        #         tau_processed[:, i] = (
        #             self.identif_config["reduction_ratio"][joint_name]
        #             * self.identif_config["kmotor"][joint_name]
        #             * self.raw_data["torques"][:, i]
        #             + 9.81 * self.robot.data.mass[
        #                 self.robot.model.getJointId(joint_name)
        #             ]
        #         )
        #     else:
        #         tau_processed[:, i] = (
        #             self.identif_config["reduction_ratio"][joint_name]
        #             * self.identif_config["kmotor"][joint_name]
        #             * self.raw_data["torques"][:, i]
        #         )
        self.processed_data["torques"] = tau_processed
        return self.processed_data["torques"]
    
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
        Dynamically determines the optimization bounds for a list of variable strings.

        The logic is designed to be highly conservative, prioritizing terms that can be 
        negative or zero (N_BOUND) based on physical meaning or algebraic structure.

        Rules for N_BOUND:
        1. If the string contains keywords for center-of-mass products (mx, my, mz), offsets (off), 
        or cross-inertia terms (Ixy, Ixz, Iyz).
        2. If the string contains a principal moment of inertia (Ixx, Iyy, Izz) AND a subtraction 
        sign ('-'), indicating a difference of positive quantities which can be negative.

        Otherwise, the variable is assumed to be a non-negative physical quantity (P_BOUND).
        """

        # Keywords that indicate the variable is physically unbounded (N_BOUND)
        physical_unbounded_keywords = ['mx', 'my', 'mz', 'off', 'Ixy', 'Ixz', 'Iyz']
        
        # Keywords for principal moments of inertia (used for the subtraction check)
        principal_inertia_keywords = ['Ixx', 'Iyy', 'Izz']

        dynamic_bounds = []
        
        for var_string in variable_list:
            is_unbounded = False
            
            # Rule 1: Check for physically unbounded terms (center-of-mass, cross-inertia, offset)
            for keyword in physical_unbounded_keywords:
                if keyword in var_string:
                    is_unbounded = True
                    break
            
            # Rule 2: Check for algebraic difference of principal inertias
            # This addresses the user's point about the minus sign leading to negative values.
            if not is_unbounded and '-' in var_string:
                for keyword in principal_inertia_keywords:
                    if keyword in var_string:
                        is_unbounded = True
                        break

            if is_unbounded:
                # Assign N_BOUND if any condition for unboundedness is met
                dynamic_bounds.append(self.N_BOUND)
            else:
                # Otherwise, assign P_BOUND (e.g., friction, actuator inertia, or positive sums)
                dynamic_bounds.append(self.P_BOUND)

        return dynamic_bounds

