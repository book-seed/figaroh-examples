#!/usr/bin/env python3

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
TALOS torso-arm calibration using the FIGAROH framework.

This script demonstrates calibration of the TALOS humanoid robot's
torso-arm kinematic chain using experimental data.
"""

import logging
import sys
from os.path import dirname, join, abspath
from pathlib import Path

import numpy as np

# Configure logging at application entry point
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(name)s - %(levelname)s - %(message)s",
)

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from figaroh.tools.robot import load_robot
from examples.talos.utils.talos_tools import TALOSCalibration


def main(visualization=True, verbose=True):
    """
    Main function for TALOS torso-arm calibration.
    
    Args:
        visualization (bool): Whether to show visualization plots
        verbose (bool): Whether to enable verbose output
    """
    print("=" * 60)
    print("TALOS Torso-Arm Calibration with FIGAROH")
    print("=" * 60)
        
    # Load robot
    robot = load_robot(
        "urdf/talos_full_v2.urdf",
        package_dirs="../../models",
        load_by_urdf=True,
    )
    
    # Initialize calibration
    print("\nInitializing TALOS calibration...")
    calibration = TALOSCalibration(
        robot=robot,
        config_file="config/talos_unified_config.yaml"
    )
    
    # Set required parameters that aren't in config file
    calibration.calib_config["known_baseframe"] = False
    calibration.calib_config["known_tipframe"] = False
    calibration.initialize()
    
    # Run calibration
    print("\n" + "=" * 40)
    print("Running calibration optimization...")
    print("=" * 40)
    
    result = calibration.solve(
        method="lm",
        max_iterations=3,
        outlier_threshold=3.0,
        enable_logging=verbose,
        plotting=True
    )
    
    # Save calibration results for use by update_model.py
    np.savez(
        join(dirname(abspath(__file__)), "data", "calibration_results.npz"),
        result=result.x,
    )
    print(f"Calibration results saved to data/calibration_results.npz")
    
    # Display calibration parameters
    param_count = len(calibration.calib_config['param_name'])
    print(f"\nCalibration parameters ({param_count} total):")
    for i, param_name in enumerate(calibration.calib_config['param_name']):
        print(f"  {i:2d}: {param_name}")


if __name__ == "__main__":
    try:
        main(
            visualization=True,
            verbose=False
        )
        
        print("\nCalibration completed successfully!")
        
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
