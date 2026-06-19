# TIAGo Configuration Files

## Active Configurations (in this directory)

| File | Purpose | Format |
|------|---------|--------|
| `tiago_unified_config.yaml` | **Main unified config** — used by all scripts by default | Unified (extends: `../../templates/humanoid_robot.yaml`) |
| `tiago_config_hey5.yaml` | Legacy config for TIAGo with Hey5 hand | Legacy flat format |
| `tiago_config.yaml` | Legacy config for TIAGo with Schunk hand | Legacy flat format |

## Archived Configurations (in `archive/`)

| File | Reason |
|------|--------|
| `tiago_config_mocap.yaml` | Broken — `data_file` and `sample_configs_file` commented out (missing data) |
| `tiago_config_mocap_vicon.yaml` | Broken — `data_file` and `sample_configs_file` commented out (missing data) |
| `tiago_config_palgripper.yaml` | Broken — `data_file` and `sample_configs_file` commented out (missing data) |
| `tiago_config_improved.yaml` | Redundant — experimental flat format with `inherit_from` (not `extends`) |
| `tiago_unified_config_with_templates.yaml` | Redundant — demo/exploratory copy of unified config |
| `tiago_unified_config_simple.yaml` | Redundant — simplified test copy of unified config |
| `templates/local_template.yaml` | Redundant — local template example, unused |

## Which Config to Use

- **All scripts** default to `config/tiago_unified_config.yaml` — this is the recommended
  configuration for TIAGo work.
- Legacy `tiago_config_hey5.yaml` and `tiago_config.yaml` are kept for backward
  compatibility with scripts that reference them explicitly.
- Do not add new configs in legacy flat format; always use the unified format with
  `extends:` template inheritance.
