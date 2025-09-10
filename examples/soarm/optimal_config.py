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
SOARM Optimal Configuration Generation Script

Generates optimal robot configurations for kinematic calibration using
D-optimal experimental design with the SOARM robot variants (SO100/SO101).
"""

import time
import sys
import os
import numpy as np

# Add the parent directory to Python path to enable proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.soarm.utils.soarm_tools import SOARMOptimalCalibration
from figaroh.tools.robot import load_robot


def main():
    """Main function for SOARM optimal configuration generation."""

    print("SOARM Optimal Configuration Generation")
    print("=" * 50)

    # 1. Load robot model
    print("\n1. Loading SOARM robot model...")

    # Load SO101 model by default (can be changed to SO100 if needed)
    robot = load_robot(
        "urdf/SO101/so101_new_calib.urdf",
        package_dirs="../../models",
        load_by_urdf=True,
    )
    print("SOARM robot model loaded successfully!")

    # 2. Create optimal calibration instance
    print("\n2. Setting up optimal calibration...")
    opt_calib = SOARMOptimalCalibration(robot, "config/soarm_config.yaml")

    calib_model = opt_calib.calib_config.get("calib_level", "N/A")
    print(f"Calibration model: {calib_model}")
    if hasattr(opt_calib, "minNbChosen"):
        print(f"Minimum configurations required: {opt_calib.minNbChosen}")

    # 3. Solve optimal configuration problem
    print("\n3. Solving optimal configuration selection...")

    start_time = time.time()

    try:
        opt_calib.solve(save_file=True)
        solve_time = time.time() - start_time
        print(f"Optimization completed in {solve_time:.2f} seconds")

        # 4. Display results
        print("\n4. Results Summary:")

        if (
            hasattr(opt_calib, "optimal_configurations")
            and "calibration_joint_configurations"
            in opt_calib.optimal_configurations
        ):
            selected_configs = opt_calib.optimal_configurations[
                "calibration_joint_configurations"
            ]
            print(f"Selected {len(selected_configs)} optimal configurations")

            total_candidates = opt_calib.calib_config.get("nb_sample", "N/A")
            if isinstance(total_candidates, int):
                print(f"Total candidates: {total_candidates}")
                ratio = len(selected_configs) / total_candidates
                print(f"Selection ratio: {ratio:.2%}")

        # 5. Show optimization quality
        if hasattr(opt_calib, "detroot_whole"):
            det_root = opt_calib.detroot_whole
            print(f"Information matrix determinant root: {det_root:.4e}")

        if hasattr(opt_calib, "optimal_weights"):
            weights = opt_calib.optimal_weights
            if hasattr(weights, "__len__") and len(weights) > 0:
                print(f"Weight sum: {np.sum(weights):.4f}")
                print(f"Non-zero weights: {np.count_nonzero(weights)}")

        # 6. Camera calibration specific information
        print("\n5. Camera Calibration Info:")
        camera_frame = opt_calib.calib_config.get("ref_frame", "N/A")
        print(f"Reference camera frame: {camera_frame}")

        markers = opt_calib.calib_config.get("markers", [])
        if markers:
            for i, marker in enumerate(markers):
                ref_joint = marker.get("ref_joint", "N/A")
                measure = marker.get("measure", [])
                marker_info = f"Joint {ref_joint}, Measurements: {measure}"
                print(f"Marker {i}: {marker_info}")

        print("\n" + "=" * 50)
        print("SOARM Optimal Configuration Generation Completed!")
        print("Results saved to 'results/' directory")

    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        print("Please check:")
        print("- Robot model file exists and is valid")
        print("- Configuration file is properly formatted")
        print("- Data file contains sufficient samples")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
