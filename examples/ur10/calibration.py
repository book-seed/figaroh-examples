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

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import UR10Calibration
from figaroh.tools.robot import load_robot


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="UR10 kinematic calibration")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ur10_unified_config.yaml",
        help="Path to unified config YAML file",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="urdf/ur10_robot.urdf",
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (INFO) logging"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main function for UR10 kinematic calibration."""
    # Validate input files
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load UR10 robot model
        ur10 = load_robot(
            args.urdf,
            package_dirs="../../models",
            load_by_urdf=True,
        )

        # Create calibration object
        ur10_calib = UR10Calibration(
            robot=ur10,
            config_file=args.config,
        )

        # Set required parameters that aren't in config file
        ur10_calib.calib_config["known_baseframe"] = False
        ur10_calib.calib_config["known_tipframe"] = False

        print("UR10 Calibration object created successfully!")
        print("Ready to initialize and solve calibration...")

        # Initialize the calibration
        ur10_calib.initialize()
        print("Calibration initialized successfully!")

        # Solve the calibration
        ur10_calib.solve(plotting=True, enable_logging=False)
        print("Calibration solved successfully!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(args)
