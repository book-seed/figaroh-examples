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

import sys
import os

# Add the parent directory to Python path to enable proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from examples.h1v2.utils.h1v2_tools import H1v2Identification
from figaroh.tools.robot import load_robot


def main():
    """Main function for H1v2 dynamic parameter identification."""
    # Load robot model
    h1v2 = load_robot( 
        "urdf/h1_2_handless.urdf",
        load_by_urdf=True,
        # robot_pkg="h12"
    )

    # Create identification object
    h1v2_iden = H1v2Identification(h1v2, "config/h1v2_config.yaml")

    # Define additional parameters excluded from yaml files
    ps = h1v2_iden.identif_config

    ps["kmotor"] = {
        'left_hip_yaw_joint': 1.0, 
        'right_hip_yaw_joint': 1.0, 
        'torso_joint': 1.0, 
        'left_hip_pitch_joint': 1.0, 
        'right_hip_pitch_joint': 1.0, 
        'left_shoulder_pitch_joint': 1.0, 
        'right_shoulder_pitch_joint': 1.0, 
        'left_hip_roll_joint': 1.0, 
        'right_hip_roll_joint': 1.0, 
        'left_shoulder_roll_joint': 1.0, 
        'right_shoulder_roll_joint': 1.0, 
        'left_knee_joint': 1.0, 
        'right_knee_joint': 1.0, 
        'left_shoulder_yaw_joint': 1.0, 
        'right_shoulder_yaw_joint': 1.0, 
        'left_ankle_pitch_joint': 1.0, 
        'right_ankle_pitch_joint': 1.0, 
        'left_elbow_joint': 1.0, 
        'right_elbow_joint': 1.0, 
        'left_ankle_roll_joint': 1.0, 
        'right_ankle_roll_joint': 1.0
    }
    
    ps["active_joints"] = ['left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint']

    # Joint parameters
    ps["act_Jid"] = [h1v2_iden.model.getJointId(i) for i in ps["active_joints"]]
    ps["act_J"] = [h1v2_iden.model.joints[jid] for jid in ps["act_Jid"]]
    ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
    ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]

    # Dataset paths
    ps["pos_data"] = "data/identification/h1_position.csv"
    ps["vel_data"] = "data/identification/h1_velocity.csv"
    ps["torque_data"] = "data/identification/h1_effort.csv"

    # Initialize identification process
    # Note: truncate parameter now accepts:
    # - None: no truncation
    # - (start, end): custom truncation indices
    h1v2_iden.initialize()

    # Solve identification
    h1v2_iden.solve(
        decimate=True,
        plotting=True,
        save_results=True,
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("TX40 DYNAMIC PARAMETER IDENTIFICATION RESULTS")
    print("=" * 60)

    print(
        f"Number of base parameters identified: "
        f"{len(h1v2_iden.params_base)}"
    )
    print(f"Correlation coefficient: {h1v2_iden.correlation:.4f}")

    if hasattr(h1v2_iden, 'result'):
        for key, value in h1v2_iden.result.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {type(value).__name__} of length {len(value)}")

    print("\nBase parameters:")
    for i, param_name in enumerate(h1v2_iden.params_base):
        print(f"{i + 1:2d}. {param_name}: {h1v2_iden. phi_base[i]:10.6f}")

    print("\nIdentification completed successfully!")
    return h1v2_iden


if __name__ == "__main__":
    main()
