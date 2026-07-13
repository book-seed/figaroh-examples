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

from typing import Any
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")
import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import OptimalTrajectoryIPOPT
from figaroh.tools.robot import load_robot

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="UR10 optimal trajectory generation")
    parser.add_argument("--config", type=str, default="config/ur10_unified_config.yaml", help="Path to unified config YAML file")
    parser.add_argument("--urdf", type=str, default="../../models/ur_description/urdf/ur10_robot.urdf",help="Path to robot URDF file")
    parser.add_argument("--model", type=str, default="../../models", help="Path to robot mesh file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (INFO) logging")
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """Main function for UR10 optimal trajectory generation."""
    # Validate input files
    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
        
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        ur10 = load_robot(args.urdf, package_dirs=model_path, load_by_urdf=True)
        ur10_traj = OptimalTrajectoryIPOPT(robot=ur10, config_file=args.config)

        # add attribute in ur10_traj.identif_config
        ps = ur10_traj.identif_config
        ps["act_Jid"] = [ur10_traj.model.getJointId(i) for i in ps["active_joints"]]
        ps["act_J"] = [ur10_traj.model.joints[jid] for jid in ps["act_Jid"]]
        ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
        ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]

        ur10_traj.initialize()
        optimal_trajectory = ur10_traj.solve(stack_reps=2)

        if optimal_trajectory is not None:
            print("Optimal trajectory generation completed successfully!")
            ur10_traj.save_results()
            ur10_traj.plot_results()
        else:
            print("Failed to generate optimal trajectory. Check constraints and parameters.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(args)
