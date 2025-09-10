# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SOARM Robot Tools

This module provides specialized calibration and identification tools for
SOARM robots (SO100 and SO101 variants), including camera-based calibration
using ArUco markers and optimal configuration generation.
"""

import os
import numpy as np
import yaml
from scipy.optimize import least_squares
from typing import Dict, Any, List

# Import FIGAROH modules
from figaroh.calibration.calibration_tools import (
    calc_updated_fkm,
    get_LMvariables,
)
from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.optimal.base_optimal_calibration import BaseOptimalCalibration
from figaroh.utils.error_handling import CalibrationError


class SOARMCalibration(BaseCalibration):
    """
    SOARM robot calibration class for kinematic parameter estimation
    using camera-based measurements and ArUco markers.

    Supports both SO100 and SO101 robot variants with comprehensive
    camera calibration pipeline and parameter optimization.
    """

    def __init__(self, robot, config_file: str, del_list: List = None):
        """
        Initialize SOARM calibration.

        Args:
            robot: Robot model loaded via FIGAROH
            config_file: Path to YAML configuration file
            del_list: List of samples to exclude from calibration
        """
        super().__init__(robot, config_file, del_list or [])

        # SOARM-specific initialization
        self.camera_frame = None
        self.marker_positions = {}

        # Validate SOARM-specific configuration
        self._validate_soarm_config()

    def _validate_soarm_config(self):
        """Validate SOARM-specific configuration parameters."""
        required_params = ["base_frame", "tool_frame", "markers"]

        for param in required_params:
            if param not in self.calib_config:
                raise CalibrationError(f"Missing required parameter: {param}")

        # Validate marker configuration
        markers = self.calib_config.get("markers", [])
        if not markers:
            raise CalibrationError("At least one marker must be configured")

        for marker in markers:
            if "ref_joint" not in marker or "measure" not in marker:
                raise CalibrationError(
                    "Each marker must have 'ref_joint' and 'measure' fields"
                )

    def cost_function(self, var: np.ndarray) -> np.ndarray:
        """
        Cost function for SOARM calibration optimization.

        Args:
            var: Optimization variables (kinematic parameters)

        Returns:
            Residual vector for least squares optimization
        """
        coeff_ = self.calib_config.get("coeff_regularize", 0.001)

        # Calculate forward kinematics with updated parameters
        PEEe = calc_updated_fkm(
            self.model, self.data, var, self.q_measured, self.calib_config
        )

        # Compute residuals
        measurement_residuals = self.PEE_measured - PEEe

        # Add regularization term for parameter stability
        nb_markers = self.calib_config.get("NbMarkers", 1)
        calib_index = self.calib_config.get("calibration_index", 6)
        param_start = 6
        param_end = -nb_markers * calib_index if nb_markers > 0 else len(var)

        if param_end < param_start:
            param_end = len(var)

        regularization_residuals = np.sqrt(coeff_) * var[param_start:param_end]

        # Combine residuals
        res_vect = np.append(measurement_residuals, regularization_residuals)
        return res_vect

    def solve_optimization(self) -> Dict[str, Any]:
        """
        Solve the SOARM calibration optimization problem.

        Returns:
            Dictionary containing optimization results
        """
        try:
            # Get initial parameter values and bounds
            var_0, bounds_low, bounds_up = get_LMvariables(
                self.model, self.calib_config
            )

            # Set initial tip pose if provided
            tip_pose = self.calib_config.get("tip_pose", [0, 0, 0, 0, 0, 0])
            if len(tip_pose) == 6:
                calib_index = self.calib_config.get("calibration_index", 6)
                var_0[-calib_index:] = np.array(tip_pose)[:calib_index]

            # Perform least squares optimization
            bounds = (bounds_low, bounds_up)

            result = least_squares(
                self.cost_function,
                var_0,
                bounds=bounds,
                method="trf",  # Trust Region Reflective algorithm
                verbose=2 if self.calib_config.get("verbose", False) else 0,
            )

            # Process results
            param_names = self.calib_config.get("param_name", [])
            param_values = result.x[: len(param_names)]

            self.calibrated_param = dict(zip(param_names, param_values))

            # Store results
            self.results = {
                "success": result.success,
                "residual_norm": np.linalg.norm(result.fun),
                "iterations": result.nfev,
                "calibrated_parameters": self.calibrated_param,
                "optimization_result": result,
            }

            return self.results

        except Exception as e:
            raise CalibrationError(f"Optimization failed: {str(e)}")

    def validate_results(
        self, outlier_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Validate calibration results and detect outliers.

        Args:
            outlier_threshold: Threshold for outlier detection (meters)

        Returns:
            Validation results dictionary
        """
        if outlier_threshold is None:
            outlier_threshold = self.calib_config.get("outlier_eps", 0.02)

        # Calculate final residuals
        final_var = self.results["optimization_result"].x
        final_residuals = self.cost_function(final_var)

        # Detect outliers in position measurements
        nb_samples = self.calib_config.get("NbSample", 0)
        nb_markers = self.calib_config.get("NbMarkers", 1)

        outlier_indices = []

        if nb_samples > 0 and nb_markers > 0:
            # Reshape residuals to per-sample format
            residuals_reshaped = final_residuals[: nb_samples * nb_markers * 3]
            residuals_reshaped = residuals_reshaped.reshape((nb_samples, -1))

            # Calculate per-sample residual norms
            sample_norms = np.linalg.norm(residuals_reshaped, axis=1)

            # Identify outliers
            outlier_indices = np.where(sample_norms > outlier_threshold)[0]

        validation_results = {
            "outlier_threshold": outlier_threshold,
            "outlier_indices": outlier_indices.tolist(),
            "num_outliers": len(outlier_indices),
            "max_residual": np.max(np.abs(final_residuals)),
            "mean_residual": np.mean(np.abs(final_residuals)),
            "residual_std": np.std(final_residuals),
        }

        return validation_results


class SOARMOptimalCalibration(BaseOptimalCalibration):
    """
    SOARM optimal calibration configuration generator.

    Uses D-optimal experimental design to select the best robot
    configurations for kinematic calibration with camera measurements.
    """

    def __init__(self, robot, config_file: str):
        """
        Initialize SOARM optimal calibration.

        Args:
            robot: Robot model loaded via FIGAROH
            config_file: Path to YAML configuration file
        """
        super().__init__(robot, config_file)

        # SOARM-specific initialization
        self.camera_configurations = []
        self.marker_visibility = {}

        # Initialize camera-specific parameters
        self._setup_camera_parameters()

    def _setup_camera_parameters(self):
        """Setup camera-specific parameters for optimal calibration."""
        # Camera pose and constraints
        self.camera_pose = self.calib_config.get(
            "camera_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        # Marker visibility constraints
        markers = self.calib_config.get("markers", [])
        for marker in markers:
            ref_joint = marker.get("ref_joint", 0)
            measures = marker.get("measure", [True] * 6)
            self.marker_visibility[ref_joint] = measures

    def generate_candidate_configurations(self) -> np.ndarray:
        """
        Generate candidate robot configurations for calibration.

        Returns:
            Array of candidate joint configurations
        """
        nb_samples = self.calib_config.get("nb_sample", 100)

        # Get joint limits from robot model
        joint_limits = []
        for i in range(self.model.nq):
            if i < len(self.model.lowerPositionLimit):
                lower = self.model.lowerPositionLimit[i]
                upper = self.model.upperPositionLimit[i]
            else:
                # Default limits if not specified
                lower, upper = -np.pi, np.pi
            joint_limits.append((lower, upper))

        # Generate random configurations within joint limits
        configurations = []

        for _ in range(nb_samples):
            config = []
            for lower, upper in joint_limits:
                # Generate random value within joint limits
                value = np.random.uniform(lower, upper)
                config.append(value)
            configurations.append(config)

        return np.array(configurations)

    def evaluate_configuration_quality(
        self, configurations: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the quality of robot configurations for calibration.

        Args:
            configurations: Array of joint configurations

        Returns:
            Quality scores for each configuration
        """
        scores = []

        for config in configurations:
            # Calculate manipulability and observability metrics
            # This is a simplified version - in practice, you would
            # evaluate the information matrix condition number

            # Basic manipulability measure
            joint_range_score = np.mean(np.abs(config))

            # Distance from joint limits (prefer middle ranges)
            limit_penalty = 0.0
            for i, q in enumerate(config):
                if i < len(self.model.lowerPositionLimit):
                    lower = self.model.lowerPositionLimit[i]
                    upper = self.model.upperPositionLimit[i]

                    # Penalty for being too close to limits
                    range_width = upper - lower
                    normalized_q = (q - lower) / range_width

                    # Penalty increases near limits (0 or 1)
                    limit_penalty += min(normalized_q, 1 - normalized_q)

            # Combine scores (higher is better)
            total_score = joint_range_score + limit_penalty
            scores.append(total_score)

        return np.array(scores)


def save_soarm_calibration_results(
    soarm_calib: SOARMCalibration,
    file_path: str = None,
    file_format: str = "yaml",
) -> str:
    """
    Save SOARM calibration results to file.

    Args:
        soarm_calib: SOARM calibration object with results
        file_path: Output file path (auto-generated if None)
        file_format: Output format ('yaml' or 'json')

    Returns:
        Path to saved file
    """
    if file_path is None:
        nb_samples = soarm_calib.calib_config.get("NbSample", "unknown")
        file_path = (
            f"results/soarm_calibration_results_{nb_samples}.{file_format}"
        )

    # Ensure results directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Prepare results data
    results_data = {
        "calibration_info": {
            "robot_type": "SOARM",
            "base_frame": soarm_calib.calib_config.get("base_frame"),
            "tool_frame": soarm_calib.calib_config.get("tool_frame"),
            "nb_samples": soarm_calib.calib_config.get("NbSample"),
            "calib_level": soarm_calib.calib_config.get("calib_level"),
        },
        "calibrated_parameters": soarm_calib.calibrated_param,
        "optimization_summary": {
            "success": soarm_calib.results.get("success"),
            "residual_norm": float(
                soarm_calib.results.get("residual_norm", 0)
            ),
            "iterations": soarm_calib.results.get("iterations"),
        },
    }

    # Add validation results if available
    if hasattr(soarm_calib, "validation_results"):
        results_data["validation"] = soarm_calib.validation_results

    # Save to file
    with open(file_path, "w") as f:
        if file_format.lower() == "yaml":
            yaml.dump(
                results_data, f, default_flow_style=False, sort_keys=False
            )
        else:  # JSON format
            import json

            json.dump(results_data, f, indent=2)

    return file_path


def load_soarm_robot(
    variant: str = "SO101", package_dirs: str = "../../models"
) -> Any:
    """
    Convenience function to load SOARM robot models.

    Args:
        variant: Robot variant ('SO100' or 'SO101')
        package_dirs: Path to robot model packages

    Returns:
        Loaded robot model
    """
    from figaroh.tools.robot import load_robot

    urdf_files = {
        "SO100": "urdf/SO100/so100.urdf",
        "SO101": "urdf/SO101/so101_new_calib.urdf",
    }

    if variant not in urdf_files:
        raise ValueError(
            f"Unknown SOARM variant: {variant}. "
            f"Available: {list(urdf_files.keys())}"
        )

    urdf_file = urdf_files[variant]

    return load_robot(urdf_file, package_dirs=package_dirs, load_by_urdf=True)


if __name__ == "__main__":
    # Example usage
    print("SOARM Tools - Example Usage")

    # Load robot
    soarm = load_soarm_robot("SO101")
    print(f"Loaded SOARM robot with {soarm.nq} joints")

    # Create calibration object
    soarm_calib = SOARMCalibration(soarm, "config/soarm_config.yaml")
    print("SOARM calibration object created successfully!")
