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
SOARM Robot Calibration Script

Performs kinematic calibration for SOARM robots (SO100/SO101) using
camera-based measurements and ArUco markers for parameter estimation.
"""

import sys
import os

# Add the parent directory to Python path to enable proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.soarm.utils.soarm_tools import SOARMCalibration
from figaroh.tools.robot import load_robot


def main():
    """Main function for SOARM kinematic calibration."""

    print("SOARM Robot Calibration")
    print("=" * 40)

    # 1. Load SOARM robot model
    print("\n1. Loading SOARM robot model...")

    # Load SO101 model by default (can be changed to SO100 if needed)
    soarm = load_robot(
        "urdf/SO101/so101_new_calib.urdf",
        package_dirs="../../models",
        load_by_urdf=True,
    )
    print("SOARM robot model loaded successfully!")

    # 2. Create calibration object
    print("\n2. Setting up calibration...")
    soarm_calib = SOARMCalibration(
        robot=soarm, config_file="config/soarm_config.yaml"
    )

    # Set required parameters that aren't in config file
    soarm_calib.calib_config["known_baseframe"] = False
    soarm_calib.calib_config["known_tipframe"] = False

    print("SOARM Calibration object created successfully!")
    print(f"Base frame: {soarm_calib.calib_config.get('base_frame', 'N/A')}")
    print(f"Tool frame: {soarm_calib.calib_config.get('tool_frame', 'N/A')}")

    # 3. Initialize the calibration
    print("\n3. Initializing calibration...")
    soarm_calib.initialize()

    if (
        hasattr(soarm_calib, "calib_config")
        and "param_name" in soarm_calib.calib_config
    ):
        param_names = soarm_calib.calib_config["param_name"]
        print(f"Parameters to calibrate: {param_names}")

    print("Calibration initialized successfully!")

    # 4. Solve the calibration
    print("\n4. Solving calibration problem...")
    soarm_calib.solve(plotting=True, enable_logging=True)
    print("Calibration solved successfully!")

    # 5. Display results summary
    print("\n5. Results Summary:")
    if hasattr(soarm_calib, "results"):
        results = soarm_calib.results
        if "residual_norm" in results:
            print(f"Final residual norm: {results['residual_norm']:.6f}")
        if "calibrated_parameters" in results:
            num_params = len(results["calibrated_parameters"])
            print(f"Number of calibrated parameters: {num_params}")

    print("\n" + "=" * 40)
    print("SOARM Calibration Completed!")


if __name__ == "__main__":
    main()
