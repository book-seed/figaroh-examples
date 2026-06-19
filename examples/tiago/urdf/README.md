# TIAGo URDF Files

## Symlinked (from `models/tiago_description/robots/`)

None — all TIAGo URDFs in this directory are kept as copies.

## Kept as copies (cannot symlink)

| File | models/ counterpart | Reason |
|---|---|---|
| `tiago.urdf` | `models/tiago_description/robots/tiago.urdf` | Content differs: examples copy has extra `<material>` definitions (Black, FlatBlack, Orange) and explicit `<inertial>` elements on `base_footprint` and `hand_grasping_frame` |
| `tiago_no_hand.urdf` | `models/tiago_description/robots/tiago_no_hand.urdf` | Content differs: examples copy has extra `<material>` definitions and explicit `<inertial>` on `base_footprint` |
| `tiago_48_hey5.urdf` | No matching file | Variant URDF (48cm torso + Hey5 hand) only used by figaroh-examples |
| `tiago_48_schunk.urdf` | No matching file | Variant URDF (48cm torso + Schunk hand) only used by figaroh-examples |
| `tiago_hey5.urdf` | No matching file | Variant URDF (standard torso + Hey5 hand) only used by figaroh-examples |
| `tiago_palgripper.urdf` | No matching file | Variant URDF (PalGripper end-effector) only used by figaroh-examples |
| `tiago_schunk.urdf` | No matching file | Variant URDF (standard torso + Schunk hand) only used by figaroh-examples |

**Note:** If the `models/tiago_description/robots/` package is updated upstream, consider
porting the extra material definitions and inertial elements to keep these copies in sync.
