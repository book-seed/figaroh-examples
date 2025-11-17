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
    # TODO: these bounds are generated bby AI, need to be double checked
    P_BOUND = (0, None)   
    N_BOUND = (None, None) 

    bounds = (
        N_BOUND, # 1. Izz_left_hip_yaw_joint + ... + 0.151*my_left_hip_pitch_joint + ... (Contains my, so N)
        N_BOUND, # 2. mx_left_hip_pitch_joint + 1.0*mx_left_hip_roll_joint (mx is N)
        N_BOUND, # 3. mz_left_hip_pitch_joint (mz is N)
        N_BOUND, # 4. Ixx_left_hip_pitch_joint - 1.0*Izz_left_hip_pitch_joint - ... (Difference of I, so N)
        N_BOUND, # 5. Ixy_left_hip_pitch_joint (Cross-inertia is N)
        P_BOUND, # 6. Iyy_left_hip_pitch_joint + ... + 1.0*Ia_left_hip_pitch_joint (Positive sum of Iyy, Izz, Ia)
        N_BOUND, # 7. Ixz_left_hip_pitch_joint (Cross-inertia is N)
        N_BOUND, # 8. Iyz_left_hip_pitch_joint (Cross-inertia is N)
        N_BOUND, # 9. my_left_hip_roll_joint + ... (my is N)
        N_BOUND, # 10. mz_left_hip_roll_joint - ... (mz/m mix is N)
        P_BOUND, # 11. Ixx_left_hip_roll_joint + ... (Positive sum of Ixx, m, Izz)
        N_BOUND, # 12. Ixy_left_hip_roll_joint (Cross-inertia is N)
        N_BOUND, # 13. Iyy_left_hip_roll_joint - 1.0*Izz_left_hip_roll_joint + ... (Difference of I, so N)
        N_BOUND, # 14. Ixz_left_hip_roll_joint (Cross-inertia is N)
        N_BOUND, # 15. Iyz_left_hip_roll_joint + ... (Iyz/my mix is N)
        N_BOUND, # 16. mx_left_knee_joint (mx is N)
        N_BOUND, # 17. mz_left_knee_joint - ... (mz/m mix is N)
        N_BOUND, # 18. Ixx_left_knee_joint - 1.0*Izz_left_knee_joint + ... (Difference of I, so N)
        N_BOUND, # 19. Ixy_left_knee_joint (Cross-inertia is N)
        P_BOUND, # 20. Iyy_left_knee_joint + ... (Positive sum of Iyy, m)
        N_BOUND, # 21. Ixz_left_knee_joint (Cross-inertia is N)
        N_BOUND, # 22. Iyz_left_knee_joint + ... (Iyz/my mix is N)
        N_BOUND, # 23. mx_left_ankle_pitch_joint + ... (mx is N)
        N_BOUND, # 24. mz_left_ankle_pitch_joint - ... (mz/m mix is N)
        N_BOUND, # 25. Ixx_left_ankle_pitch_joint - 1.0*Izz_left_ankle_pitch_joint + ... (Difference of I, so N)
        N_BOUND, # 26. Ixy_left_ankle_pitch_joint (Cross-inertia is N)
        P_BOUND, # 27. Iyy_left_ankle_pitch_joint + ... (Positive sum of Iyy, m, Izz)
        N_BOUND, # 28. Ixz_left_ankle_pitch_joint + 0.02*mx_left_ankle_roll_joint (Ixz/mx mix is N)
        N_BOUND, # 29. Iyz_left_ankle_pitch_joint (Cross-inertia is N)
        N_BOUND, # 30. my_left_ankle_roll_joint (my is N)
        N_BOUND, # 31. mz_left_ankle_roll_joint (mz is N)
        P_BOUND, # 32. Ixx_left_ankle_roll_joint (Ixx is P)
        N_BOUND, # 33. Ixy_left_ankle_roll_joint (Cross-inertia is N)
        N_BOUND, # 34. Iyy_left_ankle_roll_joint - 1.0*Izz_left_ankle_roll_joint (Difference of I, so N)
        N_BOUND, # 35. Ixz_left_ankle_roll_joint (Cross-inertia is N)
        N_BOUND, # 36. Iyz_left_ankle_roll_joint (Cross-inertia is N)

        # Right Leg (37-72)
        N_BOUND, # 37. Izz_right_hip_yaw_joint + ... - 0.151*my_right_hip_pitch_joint + ... (Contains my, so N)
        N_BOUND, # 38. mx_right_hip_pitch_joint + ... (mx is N)
        N_BOUND, # 39. mz_right_hip_pitch_joint (mz is N)
        N_BOUND, # 40. Ixx_right_hip_pitch_joint - 1.0*Izz_right_hip_pitch_joint - ... (Difference of I, so N)
        N_BOUND, # 41. Ixy_right_hip_pitch_joint (Cross-inertia is N)
        P_BOUND, # 42. Iyy_right_hip_pitch_joint + ... + 1.0*Ia_right_hip_pitch_joint (Positive sum)
        N_BOUND, # 43. Ixz_right_hip_pitch_joint (Cross-inertia is N)
        N_BOUND, # 44. Iyz_right_hip_pitch_joint (Cross-inertia is N)
        N_BOUND, # 45. my_right_hip_roll_joint + ... (my is N)
        N_BOUND, # 46. mz_right_hip_roll_joint - ... (mz/m mix is N)
        P_BOUND, # 47. Ixx_right_hip_roll_joint + ... (Positive sum)
        N_BOUND, # 48. Ixy_right_hip_roll_joint (Cross-inertia is N)
        N_BOUND, # 49. Iyy_right_hip_roll_joint - 1.0*Izz_right_hip_roll_joint + ... (Difference of I, so N)
        N_BOUND, # 50. Ixz_right_hip_roll_joint (Cross-inertia is N)
        N_BOUND, # 51. Iyz_right_hip_roll_joint + ... (Iyz/my mix is N)
        N_BOUND, # 52. mx_right_knee_joint (mx is N)
        N_BOUND, # 53. mz_right_knee_joint - ... (mz/m mix is N)
        N_BOUND, # 54. Ixx_right_knee_joint - 1.0*Izz_right_knee_joint + ... (Difference of I, so N)
        N_BOUND, # 55. Ixy_right_knee_joint (Cross-inertia is N)
        P_BOUND, # 56. Iyy_right_knee_joint + ... (Positive sum)
        N_BOUND, # 57. Ixz_right_knee_joint (Cross-inertia is N)
        N_BOUND, # 58. Iyz_right_knee_joint + ... (Iyz/my mix is N)
        N_BOUND, # 59. mx_right_ankle_pitch_joint + ... (mx is N)
        N_BOUND, # 60. mz_right_ankle_pitch_joint - ... (mz/m mix is N)
        N_BOUND, # 61. Ixx_right_ankle_pitch_joint - 1.0*Izz_right_ankle_pitch_joint + ... (Difference of I, so N)
        N_BOUND, # 62. Ixy_right_ankle_pitch_joint (Cross-inertia is N)
        P_BOUND, # 63. Iyy_right_ankle_pitch_joint + ... (Positive sum)
        N_BOUND, # 64. Ixz_right_ankle_pitch_joint + 0.02*mx_right_ankle_roll_joint (Ixz/mx mix is N)
        N_BOUND, # 65. Iyz_right_ankle_pitch_joint (Cross-inertia is N)
        N_BOUND, # 66. my_right_ankle_roll_joint (my is N)
        N_BOUND, # 67. mz_right_ankle_roll_joint (mz is N)
        P_BOUND, # 68. Ixx_right_ankle_roll_joint (Ixx is P)
        N_BOUND, # 69. Ixy_right_ankle_roll_joint (Cross-inertia is N)
        N_BOUND, # 70. Iyy_right_ankle_roll_joint - 1.0*Izz_right_ankle_roll_joint (Difference of I, so N)
        N_BOUND, # 71. Ixz_right_ankle_roll_joint (Cross-inertia is N)
        N_BOUND, # 72. Iyz_right_ankle_roll_joint (Cross-inertia is N)

        # TORSO & ARM PARAMETERS - (73-129)
        N_BOUND, # 73. Izz_torso_joint + ... + 0.28603*my_left_shoulder_pitch_joint + ... (Contains my and mx, so N)
        N_BOUND, # 74. mx_left_shoulder_pitch_joint + ... (mx/m mix is N)
        N_BOUND, # 75. mz_left_shoulder_pitch_joint - ... (mz/m mix is N)
        N_BOUND, # 76. Ixx_left_shoulder_pitch_joint - 1.0*Izz_left_shoulder_pitch_joint - ... (Difference of I, so N)
        N_BOUND, # 77. Ixy_left_shoulder_pitch_joint - ... (Ixy/m/mx mix is N)
        N_BOUND, # 78. Iyy_left_shoulder_pitch_joint + ... + 0.0684*mx_left_shoulder_roll_joint + ... (Contains mx, so N)
        N_BOUND, # 79. Ixz_left_shoulder_pitch_joint + ... (Ixz/m/mx mix is N)
        N_BOUND, # 80. Iyz_left_shoulder_pitch_joint + ... (Iyz/m mix is N)
        N_BOUND, # 81. my_left_shoulder_roll_joint (my is N)
        N_BOUND, # 82. mz_left_shoulder_roll_joint - ... (mz/m mix is N)
        N_BOUND, # 83. Ixx_left_shoulder_roll_joint + ... - 0.2912*mz_left_shoulder_yaw_joint + ... (Contains mz, so N)
        N_BOUND, # 84. Ixy_left_shoulder_roll_joint (Cross-inertia is N)
        N_BOUND, # 85. Iyy_left_shoulder_roll_joint - 1.0*Izz_left_shoulder_roll_joint + ... (Difference of I, so N)
        N_BOUND, # 86. Ixz_left_shoulder_roll_joint - ... (Ixz/m/mz mix is N)
        N_BOUND, # 87. Iyz_left_shoulder_roll_joint (Cross-inertia is N)
        N_BOUND, # 88. mx_left_shoulder_yaw_joint + ... (mx/m mix is N)
        N_BOUND, # 89. my_left_shoulder_yaw_joint + ... (my/m mix is N)
        N_BOUND, # 90. Ixx_left_shoulder_yaw_joint - 1.0*Iyy_left_shoulder_yaw_joint + ... (Difference of I, so N)
        N_BOUND, # 91. Ixy_left_shoulder_yaw_joint - ... (Ixy/m/my mix is N)
        N_BOUND, # 92. Ixz_left_shoulder_yaw_joint + ... (Ixz/m mix is N)
        N_BOUND, # 93. Iyz_left_shoulder_yaw_joint + ... (Iyz/m/my mix is N)
        N_BOUND, # 94. Izz_left_shoulder_yaw_joint + ... + 0.0658*my_left_elbow_joint + ... (Contains my, so N)
        N_BOUND, # 95. mx_left_elbow_joint (mx is N)
        N_BOUND, # 96. mz_left_elbow_joint (mz is N)
        N_BOUND, # 97. Ixx_left_elbow_joint - 1.0*Izz_left_elbow_joint (Difference of I, so N)
        N_BOUND, # 98. Ixy_left_elbow_joint (Cross-inertia is N)
        P_BOUND, # 99. Iyy_left_elbow_joint (Iyy is P)
        N_BOUND, # 100. Ixz_left_elbow_joint (Cross-inertia is N)
        N_BOUND, # 101. Iyz_left_elbow_joint (Cross-inertia is N)

        # Right Arm (102-129)
        N_BOUND, # 102. mx_right_shoulder_pitch_joint + ... (mx/m mix is N)
        N_BOUND, # 103. mz_right_shoulder_pitch_joint - ... (mz/m mix is N)
        N_BOUND, # 104. Ixx_right_shoulder_pitch_joint - 1.0*Izz_right_shoulder_pitch_joint - ... (Difference of I, so N)
        N_BOUND, # 105. Ixy_right_shoulder_pitch_joint + ... (Ixy/m/mx mix is N)
        N_BOUND, # 106. Iyy_right_shoulder_pitch_joint + ... + 0.0684*mx_right_shoulder_roll_joint + ... (Contains mx, so N)
        N_BOUND, # 107. Ixz_right_shoulder_pitch_joint + ... (Ixz/m/mx mix is N)
        N_BOUND, # 108. Iyz_right_shoulder_pitch_joint - ... (Iyz/m mix is N)
        N_BOUND, # 109. my_right_shoulder_roll_joint (my is N)
        N_BOUND, # 110. mz_right_shoulder_roll_joint - ... (mz/m mix is N)
        N_BOUND, # 111. Ixx_right_shoulder_roll_joint + ... - 0.2912*mz_right_shoulder_yaw_joint + ... (Contains mz, so N)
        N_BOUND, # 112. Ixy_right_shoulder_roll_joint (Cross-inertia is N)
        N_BOUND, # 113. Iyy_right_shoulder_roll_joint - 1.0*Izz_right_shoulder_roll_joint + ... (Difference of I, so N)
        N_BOUND, # 114. Ixz_right_shoulder_roll_joint - ... (Ixz/m/mz mix is N)
        N_BOUND, # 115. Iyz_right_shoulder_roll_joint (Cross-inertia is N)
        N_BOUND, # 116. mx_right_shoulder_yaw_joint + ... (mx/m mix is N)
        N_BOUND, # 117. my_right_shoulder_yaw_joint - ... (my/m mix is N)
        N_BOUND, # 118. Ixx_right_shoulder_yaw_joint - 1.0*Iyy_right_shoulder_yaw_joint + ... (Difference of I, so N)
        N_BOUND, # 119. Ixy_right_shoulder_yaw_joint + ... (Ixy/m/my mix is N)
        N_BOUND, # 120. Ixz_right_shoulder_yaw_joint + ... (Ixz/m mix is N)
        N_BOUND, # 121. Iyz_right_shoulder_yaw_joint - ... (Iyz/m/my mix is N)
        N_BOUND, # 122. Izz_right_shoulder_yaw_joint + ... - 0.0658*my_right_elbow_joint + ... (Contains my, so N)
        N_BOUND, # 123. mx_right_elbow_joint (mx is N)
        N_BOUND, # 124. mz_right_elbow_joint (mz is N)
        N_BOUND, # 125. Ixx_right_elbow_joint - 1.0*Izz_right_elbow_joint (Difference of I, so N)
        N_BOUND, # 126. Ixy_right_elbow_joint (Cross-inertia is N)
        P_BOUND, # 127. Iyy_right_elbow_joint (Iyy is P)
        N_BOUND, # 128. Ixz_right_elbow_joint (Cross-inertia is N)
        N_BOUND, # 129. Iyz_right_elbow_joint (Cross-inertia is N)

        # VISCOUS FRICTION (fv) PARAMETERS - (130-150)
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Left Leg fv
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Right Leg fv
        P_BOUND, # Torso fv
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Left Arm fv
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Right Arm fv

        # STATIC FRICTION (fs) PARAMETERS - (151-171)
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Left Leg fs
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Right Leg fs
        P_BOUND, # Torso fs
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Left Arm fs
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Right Arm fs

        # ACTUATOR INERTIA (Ia) PARAMETERS (Remaining) - (172-187)
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Left Leg Ia (excluding hip_yaw/pitch)
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Right Leg Ia (excluding hip_yaw/pitch)
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Left Arm Ia
        P_BOUND, P_BOUND, P_BOUND, P_BOUND, # Right Arm Ia

        # OFFSET (off) PARAMETERS - (188-208)
        N_BOUND, N_BOUND, N_BOUND, N_BOUND, N_BOUND, N_BOUND, # Left Leg off
        N_BOUND, N_BOUND, N_BOUND, N_BOUND, N_BOUND, N_BOUND, # Right Leg off
        N_BOUND, # Torso off
        N_BOUND, N_BOUND, N_BOUND, N_BOUND, # Left Arm off
        N_BOUND, N_BOUND, N_BOUND, N_BOUND  # Right Arm off
    )

    h1v2_iden.solve_with_custom_solver(
        method="constrained",
        bounds=bounds,
        decimate=True,
        decimation_factor=10,
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

    # print("\nBase parameters:")
    # for i, param_name in enumerate(h1v2_iden.params_base):
        # print(f"{i + 1:2d}. {param_name}: {h1v2_iden. phi_base[i]:10.6f}")

    print("\nIdentification completed successfully!")
    return h1v2_iden


if __name__ == "__main__":
    main()
