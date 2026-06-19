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
Update TALOS URDF model with calibrated parameters.

This script takes the calibration results produced by
calibration_upperbody.py and writes them into offset.xacro
and offset.yaml files for use with the TALOS URDF model.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from os.path import dirname, join, abspath
from pathlib import Path
from typing import Optional

import numpy as np

# update estimated parameters to xacro file for left hand

torso_list = [0, 1, 2, 3, 4, 5]
arm1_list = [6, 7, 8, 11]
arm2_list = [13, 16]
arm3_list = [19, 22]
arm4_list = [24, 27]
arm5_list = [30, 33]
arm6_list = [36, 39]
arm7_list = [43, 46]  # include phiz7
total_list = [
    torso_list,
    arm1_list,
    arm2_list,
    arm3_list,
    arm4_list,
    arm5_list,
    arm6_list,
    arm7_list,
]

zero_list = []
for i in range(len(total_list)):
    zero_list = [*zero_list, *total_list[i]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update TALOS URDF model with calibrated parameters"
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


def update_parameters(
    model,
    res: Optional[np.ndarray] = None,
    calib_config: Optional[dict] = None,
) -> None:
    # If no res provided, load from saved calibration results
    if res is None:
        data_path = join(dirname(abspath(__file__)), "data", "calibration_results.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Calibration results not found at {data_path}. "
                "Run calibration_upperbody.py first."
            )
        data = np.load(data_path)
        res = data["result"]

    param_list = np.zeros((calib_config["NbJoint"], 6))

    # torso all zeros

    # arm 1
    param_list[1, 3] = res[6]
    param_list[1, 4] = res[7]

    # arm 2
    param_list[2, 0] = res[8]
    param_list[2, 2] = res[9]
    param_list[2, 3] = res[10]
    param_list[2, 5] = res[11]

    # arm 3
    param_list[3, 0] = res[12]
    param_list[3, 2] = res[13]
    param_list[3, 3] = res[14]
    param_list[3, 5] = res[15]

    # arm 4
    param_list[4, 1] = res[16]
    param_list[4, 2] = res[17]
    param_list[4, 4] = res[18]
    param_list[4, 5] = res[19]

    # arm 5
    param_list[5, 1] = res[20]
    param_list[5, 2] = res[21]
    param_list[5, 4] = res[22]
    param_list[5, 5] = res[23]

    # arm 6
    param_list[6, 1] = res[24]
    param_list[6, 2] = res[25]
    param_list[6, 4] = res[26]
    param_list[6, 5] = res[27]

    # arm 7
    param_list[7, 0] = res[28]
    param_list[7, 2] = res[29]
    param_list[7, 3] = res[30]
    param_list[7, 5] = res[31]

    joint_names = [name for i, name in enumerate(model.model.names)]
    offset_name = [
        "_x_offset",
        "_y_offset",
        "_z_offset",
        "_roll_offset",
        "_pitch_offset",
        "_yaw_offset",
    ]
    path_save_xacro = join(dirname(str(abspath(__file__))), f"data/offset.xacro")
    with open(path_save_xacro, "w") as output_file:
        for i in range(calib_config["NbJoint"]):
            for j in range(6):
                update_name = joint_names[i + 1] + offset_name[j]
                update_value = param_list[i, j]
                update_line = '<xacro:property name="{}" value="{}" / >'.format(
                    update_name, update_value
                )
                output_file.write(update_line)
                output_file.write("\n")
    path_save_yaml = join(dirname(str(abspath(__file__))), f"data/offset.yaml")
    with open(path_save_yaml, "w") as output_file:
        for i in range(calib_config["NbJoint"]):
            for j in range(6):
                update_name = joint_names[i + 1] + offset_name[j]
                update_value = param_list[i, j]
                update_line = "{}: {}".format(update_name, update_value)
                output_file.write(update_line)
                output_file.write("\n")


if __name__ == "__main__":
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Add project root to path for imports
    project_root = Path(__file__).parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Validate file existence
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    from figaroh.tools.robot import load_robot
    from examples.talos.utils.talos_tools import TALOSCalibration

    try:
        # Path to the directory containing this script
        script_dir = dirname(abspath(__file__))

        # Load robot model
        print("Loading robot model...")
        robot = load_robot(
            str(urdf_path),
            package_dirs="../../models",
            load_by_urdf=True,
        )

        # Load calibration configuration
        print("Loading calibration configuration...")
        calibration = TALOSCalibration(
            robot=robot,
            config_file=str(config_path),
        )

        # Update model parameters (res will be auto-loaded from file)
        print("Updating model parameters...")
        update_parameters(robot, calib_config=calibration.calib_config)
        print("Model parameters updated successfully.")
        print("  - data/offset.xacro")
        print("  - data/offset.yaml")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
