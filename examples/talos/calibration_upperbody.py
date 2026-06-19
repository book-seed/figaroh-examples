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

from __future__ import annotations

import argparse
import logging
import sys
from os.path import dirname, join, abspath
from pathlib import Path

import numpy as np

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from figaroh.tools.robot import load_robot
from examples.talos.utils.talos_tools import TALOSCalibration


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TALOS torso-arm calibration with FIGAROH"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/talos_unified_config.yaml",
        help="Path to unified config YAML file",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="urdf/talos_full_v2.urdf",
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (INFO) logging",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function for TALOS torso-arm calibration.

    Args:
        args: Parsed command-line arguments
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate file existence
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("TALOS Torso-Arm Calibration with FIGAROH")
    print("=" * 60)

    # Load robot
    robot = load_robot(
        str(urdf_path),
        package_dirs="../../models",
        load_by_urdf=True,
    )

    # Initialize calibration
    print("\nInitializing TALOS calibration...")
    calibration = TALOSCalibration(
        robot=robot,
        config_file=str(config_path),
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
        enable_logging=args.verbose,
        plotting=True,
    )

    # Save calibration results for use by update_model.py
    np.savez(
        join(dirname(abspath(__file__)), "data", "calibration_results.npz"),
        result=result.x,
    )
    print("Calibration results saved to data/calibration_results.npz")

    # Display calibration parameters
    param_count = len(calibration.calib_config["param_name"])
    print(f"\nCalibration parameters ({param_count} total):")
    for i, param_name in enumerate(calibration.calib_config["param_name"]):
        print(f"  {i:2d}: {param_name}")


if __name__ == "__main__":
    try:
        main(parse_args())
        print("\nCalibration completed successfully!")
    except Exception as e:
        print(f"\nError during calibration: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
