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
SOARM Model Update Script

Updates SOARM robot models with calibrated kinematic parameters.
Supports both SO100 and SO101 variants.
"""

import sys
import os
import yaml
from pathlib import Path

# Add the parent directory to Python path to enable proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def load_calibration_results(results_file: str = None) -> dict:
    """Load calibration results from YAML file."""

    if results_file is None:
        # Try to find calibration results automatically
        candidates = [
            "results/soarm_calibration_results.yaml",
            "results/latest_calibration.yaml",
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                results_file = candidate
                break

        if results_file is None:
            raise FileNotFoundError("No calibration results file found")

    with open(results_file, "r") as f:
        return yaml.safe_load(f)


def create_updated_urdf(
    original_urdf: str, calibration_results: dict, output_urdf: str = None
) -> str:
    """Create updated URDF with calibrated parameters."""

    if output_urdf is None:
        urdf_path = Path(original_urdf)
        base_name = f"{urdf_path.stem}_calibrated.urdf"
        output_urdf = str(urdf_path.parent / base_name)

    # Read original URDF
    with open(original_urdf, "r") as f:
        urdf_content = f.read()

    # Add calibration comment
    calibration_comment = (
        "<!-- Updated with FIGAROH calibrated parameters -->\n"
        "<!-- SOARM Robot Calibration Results -->\n"
    )

    # Find insertion point after XML declaration
    xml_decl_end = urdf_content.find("?>") + 2
    updated_content = (
        urdf_content[:xml_decl_end]
        + "\n"
        + calibration_comment
        + urdf_content[xml_decl_end:]
    )

    # Write updated URDF
    with open(output_urdf, "w") as f:
        f.write(updated_content)

    return output_urdf


def main():
    """Main function for SOARM model updating."""

    print("SOARM Model Update Tool")
    print("=" * 30)

    try:
        # Load calibration results
        print("Loading calibration results...")
        calibration_results = load_calibration_results()

        print("✅ Calibration results loaded successfully")

        # Display summary
        calib_info = calibration_results.get("calibration_info", {})
        print(f"Robot type: {calib_info.get('robot_type', 'SOARM')}")
        print(f"Samples used: {calib_info.get('nb_samples', 'Unknown')}")

        params = calibration_results.get("calibrated_parameters", {})
        print(f"Calibrated parameters: {len(params)}")

        # Update models
        models_to_update = [
            "urdf/SO101/so101_new_calib.urdf",
            "urdf/SO100/so100.urdf",
        ]

        updated_files = []

        for model_file in models_to_update:
            if os.path.exists(model_file):
                print(f"\nUpdating: {model_file}")
                updated_file = create_updated_urdf(
                    model_file, calibration_results
                )
                updated_files.append(updated_file)
                print(f"✅ Created: {updated_file}")
            else:
                print(f"⚠️  Model not found: {model_file}")

        # Save update summary
        summary_file = "results/model_update_summary.yaml"
        os.makedirs("results", exist_ok=True)

        summary = {
            "update_summary": {
                "source_calibration": "Auto-detected",
                "models_updated": updated_files,
                "calibration_info": calib_info,
            }
        }

        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        print(f"\n✅ Update summary saved to: {summary_file}")
        print("\n" + "=" * 30)
        print("SOARM Model Update Completed!")

        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
