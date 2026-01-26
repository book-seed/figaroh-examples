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
    )

    # Create identification object
    h1v2_iden = H1v2Identification(h1v2, "config/h1v2_config.yaml")
    h1v2_iden.filter_config = {'differentiation_method': 'gradient', 'filter_params': {'window_length': 18, 'polyorder': 6, 'mode': 'nearest'}} 

    # Define additional parameters excluded from yaml files
    ps = h1v2_iden.identif_config
    
    # ps["active_joints"] = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
    ps["active_joints"] = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint']
    # ps["untrustworthy_joints"] = ['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
    ps["untrustworthy_joints"] = ['']

    # Joint parameters
    ps["act_Jid"] = [h1v2_iden.model.getJointId(i) for i in ps["active_joints"]]
    ps["act_J"] = [h1v2_iden.model.joints[jid] for jid in ps["act_Jid"]]
    ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
    ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]

    # Dataset paths
    data_pathes_A = {}
    data_pathes_A["pos_data"] = "data/identification/mujoco/exp_A/h1_position.csv"
    data_pathes_A["vel_data"] = "data/identification/mujoco/exp_A/h1_velocity.csv"
    data_pathes_A["torque_data"] = "data/identification/mujoco/exp_A/h1_effort.csv"
    
    data_pathes_B = {}
    data_pathes_B["pos_data"] = "data/identification/mujoco/exp_B/h1_position.csv"
    data_pathes_B["vel_data"] = "data/identification/mujoco/exp_B/h1_velocity.csv"
    data_pathes_B["torque_data"] = "data/identification/mujoco/exp_B/h1_effort.csv"
    
    # Load data
    ps["mass_load"] = 2.549
    ps["which_body_loaded"] = h1v2_iden.model.getJointId('left_elbow_joint')
    # ps["which_body_loaded"] = h1v2_iden.model.getJointId('left_wrist_yaw_joint')

    # h1v2_iden.initialize_global(data_pathes_A, data_pathes_B, truncate_A=(9000, 18500), truncate_B=(6000, 25000))
    h1v2_iden.initialize_global(data_pathes_A, data_pathes_B, truncate_A=(3000, 8000), truncate_B=(3667, 13000))
    
    # Solve identification
    h1v2_iden.solve_global(
        plotting=True,
        save_results=True,
    )


if __name__ == "__main__":
    main()
