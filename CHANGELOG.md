# Changelog

## [Unreleased]

### Changed
- tools/step01_colmap.py: registration passes at 80 percent of extracted frames
- tools/can_block_b_check.py: the same 80 percent registration line as step01
- tests/test_step01.py: 87.2 percent registers as a pass and 79.9 percent stays a fail
- tools/block_e.py: say which of splat.ply or gaussian_in_mask.npy is missing when the mesh is not written
- tests/test_orbit_chain.py: a missing splat with the mask on disk keeps Block E not_started
- tests/test_orbit_chain.py: measured D1 steps stay unachieved and do not mark Block D partial without physics.json
- tests/test_block_d.py: refresh_block_d stays not_started until physics.json has material and mass_kg

### Fixed
- tools/orbit_chain.py: start_d2 reads OPENAI_API_KEY instead of an empty api_key override
- tools/block_d.py: a blank api_key does not discard OPENAI_API_KEY or invent a mass
- tests/test_orbit_chain.py: a missing key still skips D2, and a present key is not discarded by an empty api_key string
- tests/test_block_d.py: an empty api_key string does not discard a present OPENAI_API_KEY
- tools/orbit_chain.py: Block D stays not_started unless physics.json has material and mass_kg
- tools/block_f.py: an inspect with no telemetry says Isaac was not run; sim.reset is only a timeout
- tools/orbit_chain.py: F1 names each missing input (mesh, capture, mask, world_T, intrinsics) instead of blaming only Block C
- tests/test_block_f.py: empty telemetry is not a sim.reset hang
- tests/test_orbit_chain.py: orbit F1–F4 reasons list the missing inputs
- tools/orbit_chain.py: Block E is not_started when mesh.ply and deformable_type are missing; a one-body label is not a decomposition
- tools/orbit_chain.py: E1 names a missing splat.ply when the mask is on disk, and E2 stays unstarted until hulls are built
- tools/block_e.py: a missing mesh names splat.ply or gaussian_in_mask.npy instead of an or
- tests/test_orbit_chain.py: hulls and a recorded deformable_type are required before E2 or E4 are partial

### Added
- tools/orbit_chain.py: run can orbits 1–3 from Block B files through D, E, and F; skip C; log orbit_chain.md
- tests/test_orbit_chain.py: tmp_path chain; C skipped, no key, no splat, and no world_T stay unachieved
- tools/block_f.py: 7-DoF alignment, background orientation, scene USDZ, and Isaac inspect; a skip is not achieved
- tests/test_block_f.py: tmp_path fixtures for F1–F4; missing mesh, background, asset, and telemetry stay unachieved
- tools/can_instances.py: cluster can Gaussians from mask-id votes plus B3 features; missing hand labels keep mIoU null
- tools/block_d.py: D1.2–D3 writers; scene-mean copy, gated prior, VLM start, fuse/bake; learned_prior stays false
- tests/test_block_d.py: tmp_path fixtures for D1.2–D3; missing ply/key/checkpoint stay unstarted
- tools/can_hand_labels.py: score rendered instance masks against 20-view can hand labels; missing labels are a fail
- tests/test_can_hand_labels.py: a skipped or missing can hand label is never a pass
- tests/test_can_instances.py: one lifted mask is not itself a can instance
- tests/test_can_block_b_orbits.py: tmp_path fixtures per Block B sub-step; live scores keep M1 and M2 unmet
- tools/can_block_b_check.py: walk can orbit 2/3 Block A through B4 against source artifacts without treating a skip as a pass
- tools/can_block_b_check.py: score orbit1 the same way as orbit2/3 (codebook.json, block_b_orbit1.json)
- tools/block_e.py: can mesh, part, joint, and deformable writers; a null metres_per_unit keeps Chamfer unachieved
- tests/test_block_e.py: fixtures for the filter, Chamfer scale gate, one-body joints, and rigid route

### Changed
- tools/common/vlm.py: retry the vision POST without temperature when gpt-6-astra returns HTTP 400
- tools/can_block_b_check.py: report B2 LangSplat from features.codebook.npz keys instead of hardcoding unstarted
- tools/can_block_b_check.py: feature_field measured without gaussian_features.npy is a mismatch, not a pass
