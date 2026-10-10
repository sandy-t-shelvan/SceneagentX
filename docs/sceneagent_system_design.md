# SceneAgent Reproduction — System Design

Oct 10, 2026 · @Sandeep Shelvan

## Scope, sources and limitations

This is a reproduction design for [SceneAgent](https://computationalrobotics.seas.harvard.edu/SceneAgent/) (Hollis, Fan, Yang; Harvard Computational Robotics Group, 2026 preprint). The paper PDF and code are not yet public (the site says "coming soon"), so every block below is reverse-engineered from the project page: its prose, Figure 1 (pipeline), Figure 4 (agent skills), the 30+ video captions, and the asset credits.

What I could and could not inspect:

- Read in full via a browser session on 2026-10-10: all page text, Figure 1 (pipeline), Figure 4 (nine agent skills: SCENEAGENT-RUN-PIPELINE, -INGEST-DATA, -RUN-STEP, -ALIGN-SCENE, -ORIENT-BACKGROUND, -PACKAGE-DATA-PROCESSED, -INSPECT-ISAAC, -OBJECT-SISTERS, -SCENE-SISTERS), the section-5 physics-prior workflow diagram, the benchmark chart (section 1.4), the sister carousels, and sampled frames from every video (the 2:35 writing video and 1:40 parchment video at a few points only).
- Three things the page shows that a text fetch misses: (1) the pipeline is driven by a natural-language prompt ("convert the iPhone capture of this kitchen into a simulation scene with every door, drawer and appliance articulated"); (2) SceneAgent is packaged as skills for off-the-shelf coding agents and benchmarked with and without those skills; (3) the physics prior is a concrete per-Gaussian workflow (DINOv2 + OpenCLIP features → prior → per-Gaussian distribution → object, part, field), not only a VLM guess.
- Labelled throughout: **\[site\]** = stated or shown on the page; **\[inferred\]** = my engineering choice to fill a gap the page leaves open. Treat every \[inferred\] item as a decision to revisit once the paper is out.

The design targets the primary path (3DGS capture of a tabletop workspace, Franka Panda with UMI gripper, Isaac Lab) first; photogrammetry, LiDAR, generated-3DGS inputs and MuJoCo/Unreal exports are second-tier.

## Big picture

SceneAgent is a real-to-sim-to-real system: it turns a 3D capture of a real workspace into a physics-enabled simulation scene, mass-produces training demonstrations inside that scene, finetunes a vision-language-action (VLA) policy on them, and deploys the policy zero-shot on the real robot. \[site\]

Three ideas carry the work:

1. **Everything is per-Gaussian first.** Semantic features, object masks and physics priors are attached to the individual 3D Gaussians of a splat, then baked into object-level meshes and materials. The splat stays as the photoreal background; the meshes become the colliders and rigid/soft bodies. \[site\]
2. **Agents replace hand-cleanup.** Every conversion step (segmentation, crop/infill, physics, decomposition, alignment) is error-prone. A VLM-driven agent swarm runs each step, renders the result from inside the simulator, visually checks it, and reissues corrections (scale, rotation, placement are the stated common failures). \[site\]
3. **Digital sisters, not twins.** Each object is regenerated as a family of near-variants (geometry, texture), and whole-scene layouts and lighting are randomized, so the policy learns a distribution rather than one instance. \[site\]

End-to-end data flow (details in the flow chart and blocks below):

| Stage | Input | Output | Site evidence |
| --- | --- | --- | --- |
| 1 Capture & ingest | RGB video/images, or 3DGS .ply, photogrammetry mesh, LiDAR | Trained 3DGS scene | COLMAP + Nerfstudio splatfacto-big named |
| 2 Semantics & segmentation | 3DGS | Per-Gaussian feature codebook; foreground object Gaussian sets | GroundingDINO + SAM; LangSplat-style codebook |
| 3 Crop & infill | Scene + object sets | Background splat with holes filled; isolated object splats | "predictively infill these cropped regions" |
| 4 Predictive physics | Object splats + semantics | Friction, rigidity, mass, density; baked physics material | Per-Gaussian physics model + VLM estimate |
| 5 Decomposition & articulation | Object splat/mesh | Part meshes, joints, soft-body configs | Radio, stegosaurus, engine, cockpit, cloth/rope videos |
| 6 Sisters | Processed object | 9 variants per object; N layout variants per scene | 9 sisters shown per object |
| 7 Align, orient, package | All of the above | USDZ scene for Isaac Lab / MuJoCo / Unreal | Figure 4 skills 04-07 |
| 8 Demonstration factory | USDZ scene + task | Thousands of expert episodes with rendered observations | Scripted expert, success gate, depth compositing |
| 9 Train & deploy | Episodes | LoRA-finetuned pi0.5 / GR00T; real-robot rollouts | 10,000 episodes; Franka Panda + UMI |

The page's own tagline for the output format is a single artifact: a USDZ package per scene variant, consumable by Isaac Lab (primary), MuJoCo and Unreal Engine. \[site\]

## Engineering questions the page leaves open, and the assumptions taken

These are the questions I would ask the authors before writing a line of code. Each row records the assumption this design proceeds on and what breaks if it is wrong. Treat the table as the project's assumption register: when the paper or code appears, close rows one by one.

| # | Question | Assumption taken | Risk if wrong |
| --- | --- | --- | --- |
| 1 | How is metric scale recovered from an iPhone/COLMAP capture? | ARKit pose metadata when present; otherwise a printed AprilTag board in every capture | Mass, density, grasp widths and robot reach are all wrong; policy trained on a mis-scaled twin |
| 2 | Which coordinate frame and units does the USDZ use? | Metres, Z-up, right-handed (Isaac Lab default); MuJoCo/Unreal exports convert | Silent 90° rotations and 100× scale errors between simulators |
| 3 | How do photogrammetry meshes and LiDAR scans become 3DGS? | Render the mesh from a camera sphere and train a splat on the renders | Lower splat quality for mesh inputs; LiDAR-only (no colour) scenes need a separate texture source |
| 4 | SAM or SAM 2? Which GroundingDINO weights? | SAM 2.1 hiera-large for video-consistent masks; GroundingDINO Swin-B | Instance IDs flicker between frames and objects fragment into several instances |
| 5 | What is the per-Gaussian feature dimension and codebook size? | 3-d latent per Gaussian (LangSplat autoencoder), 256-entry codebook | Too small: objects merge; too large: feature training time and memory explode |
| 6 | What is "predictive infill" concretely? | LaMa inpainting per view + masked splat fine-tune for 500 iterations | Visible seams in the background behind lifted objects; policy sees an artefact that never occurs in reality |
| 7 | What is the "per-Gaussian predictive physics model"? Trained by the authors or reused? | Workflow is fixed by the page (DINOv2 + OpenCLIP object vector + per-Gaussian geometry → prior → p\_i); the prior's training data is not. Assume a small MLP trained on PhysXNet labels with NeRF2Physics-style VLM labels as fallback | Physics values are no better than the VLM guess alone; heterogeneous objects (hat brim vs crown) lose their variation |
| 8 | Which VLM reviews steps, and is it cloud or local? | Cloud VLM via API for development; Qwen2.5-VL-7B for the on-device variant | Review latency and cost dominate pipeline time; local model misses errors the cloud model catches |
| 9 | How are collision shapes generated? | CoACD convex decomposition per part, max 32 hulls | Objects interpenetrate or rest on invisible bumps; grasps succeed in sim and fail in reality |
| 10 | How are joints proposed for articulated objects? | VLM names the moving part and joint type; axis from part geometry; validated by actuating in Isaac Lab | Radio door opens the wrong way; knob slides instead of turning |
| 11 | Which deformable solver? | PhysX cloth/FEM inside Isaac Lab for the primary path; MuJoCo flex only for the secondary export | Soft-body parameters do not transfer between PhysX and MuJoCo; deformable behaviour differs per simulator |
| 12 | Which generative 3D model produces sisters, and how is in-class fidelity enforced? | TRELLIS image-to-3D with VLM-written prompts; CLIP similarity gate 0.80–0.95 to the original | Sisters drift out of class (a teacup becomes a bowl) or are near-duplicates that add nothing |
| 13 | What are the three alignment residuals weighted? | Photometric 1.0, silhouette IoU 2.0, edge 0.5; optimised with Adam over a 7-DoF similarity transform | Alignment converges to a locally plausible but wrong pose; inspection catches some but not all |
| 14 | How does the robot camera see the splat background in Isaac Lab? | It does not; frames are composited outside Isaac from a gsplat render + Isaac RTX depth/colour | Any design that renders the splat in Isaac's viewport fails; the compositor becomes a core component |
| 15 | How is lighting matched between the RTX foreground and the splat background? | Dome light estimated from the splat, shadow catcher on the table proxy, colour-matching pass | Foreground looks pasted on; the policy learns a sim-only visual cue |
| 16 | What action space does the scripted expert record? | End-effector delta pose + gripper command at 10 Hz, matching the DROID/UMI controller | Finetuned VLA emits actions the real controller cannot execute |
| 17 | What cameras are recorded? | Wrist camera (shown in the evaluation video) plus one third-person camera at the DROID position, 224×224 after resize | Train/eval camera mismatch; policy ignores the view it needs |
| 18 | Which dataset format? | LeRobot v2 (parquet + mp4), the input format openpi and Isaac-GR00T both accept | Conversion scripts per model; two models train on subtly different data |
| 19 | How many episodes, and what success-gate tolerance? | 10,000 kept episodes (the page's figure); object centroid inside the container AABB and at rest for 0.5 s | Too loose: failed demos poison training; too tight: factory throughput collapses |
| 20 | Which VLA checkpoints and LoRA settings? | pi0.5 base via openpi; GR00T N1.6 via Isaac-GR00T; LoRA rank 32 on attention + MLP, 20k steps | Results not comparable to the page's; compute budget off by 2–4× |
| 21 | How is the real-world trial layout defined? | Reference point P0 marked on the table; layouts sampled in sim and replayed on the real table with a printed template | Sim and real success rates are not comparable |
| 22 | What counts as a human escalation vs an agent retry? | 3 agent retries per step; then the editor opens with the failing renders and the last VLM correction | Agents loop forever or give up too early; wall-clock per scene unbounded |
| 23 | Is the pipeline per scene or per object? | Both: a scene run spawns one object sub-run per instance; objects can also be run standalone and dropped into hybrid scenes | Hybrid scenes (captured part into an Omniverse data-center scene) have no entry point |
| 24 | What GPU and how much VRAM for conversion? | One 24 GB GPU (RTX 4090/A5000) per scene; the splat, feature field and Isaac Lab are never resident at the same time | OOM on larger scenes; must sequence tools or shard |
| 25 | What is the licence posture for a release? | Permissive stack where possible; Isaac Sim, GR00T and pi0.5 weights carry their own terms and are listed in the inventory | Cannot redistribute converted scenes or finetuned weights |

## End-to-end flow chart

&#91;embedded content: SceneAgent reproduction flow · 12 steps in 3 stages, 1 review loop\]

The conversion stage runs top to bottom under the agent swarm; a failed inspection loops back to alignment with a correction. The packaged USDZ (accent) is the single hand-off into the sisters and policy stages.

## Block A — Capture and ingest to 3DGS

Goal: any supported capture ends as one canonical 3DGS scene (`scene.ply` + camera poses + metric scale). This is the Figure 4 skill SCENEAGENT-INGEST-DATA. \[site\]

&#91;embedded content: Block A flow · 4 steps, 1 retry loop\]

**Measurable outcome (M1):** `sceneagent ingest` yields PSNR above 27 dB and LPIPS below 0.20 on held-out views, and a metric scale error below 2 % against the tag board.

**Accepted inputs \[site\]:** RGB video or image sequence (iPhone capture was used for the kitchen), pre-trained 3DGS `.ply`, photogrammetry meshes (Polycam, Sketchfab), LiDAR scans (Matterport observatory), and generated splats (World Labs Marble, NVIDIA Lyra, Tencent HY-World).

**Path A1 — RGB to 3DGS \[site\]**

1. Frame extraction and blur rejection (ffmpeg; Laplacian-variance filter) \[inferred\].
2. Structure-from-motion with COLMAP: feature matching, sparse reconstruction, undistortion. \[site\]
3. Train 3DGS with Nerfstudio `splatfacto-big`. \[site\]
4. Agent review of training artifacts, including per-image colour calibration. \[site\] Implement as: render held-out views, compute PSNR/LPIPS, let the VLM flag exposure drift and re-run with appearance embeddings on.
5. Metric scale recovery \[inferred, required\]: COLMAP output is scale-free, and Block D (mass, density) and Block J (robot reach) need metres. Options, in order of preference: ARKit/Polycam pose metadata when present; a printed ArUco/AprilTag board in the capture; a known object dimension confirmed by the VLM.

**Path A2 — mesh or LiDAR to 3DGS \[inferred\]:** render the textured mesh from a sampled camera sphere and run Path A1 steps 3–4 on the renders, or initialise Gaussians directly from mesh vertices/LiDAR points. The page confirms photogrammetry and LiDAR scenes go through the same downstream pipeline but does not say how they become splats; this is the simplest route.

**Path A3 — existing `.ply`:** validate (SH degree, attribute names), normalise coordinate frame to Z-up right-handed, pass through.

**Outputs:** `scene.ply`, `cameras.json` (Nerfstudio transforms), `scale.json` (metres-per-unit + method), `ingest_report.json` (quality metrics, agent notes).

**Software:** COLMAP ≥ 3.9, Nerfstudio ≥ 1.1 with gsplat, ffmpeg, OpenCV, PyTorch + CUDA 12.x.

## Block B — Semantic features and object segmentation

Goal: every Gaussian carries a language-aligned feature, and each foreground object is a named set of Gaussian indices. The page names GroundingDINO + SAM for assignment and a LangSplat-style codebook for storage. \[site\]

&#91;embedded content: Block B flow · 4 steps, 1 review loop\]

**Measurable outcome (M2):** instance-mask mIoU above 0.80 against 20 hand-labelled views, and the instance count within ±1 of ground truth per scene.

**B1 — 2D feature extraction per training view**

1. Open-vocabulary detection with GroundingDINO (prompt list from a VLM scene description, e.g. "blender, mug, cutting board"). \[site\]
2. Mask refinement with SAM (SAM 2 for video-consistent masks across the capture is the better choice today) \[inferred\]. \[site names SAM\]
3. Per-mask CLIP embedding (LangSplat uses CLIP features from SAM masks at three granularities: whole, part, subpart). \[site: "inspired by LangSplat"\]

**B2 — Lift to 3D and build the codebook**

1. Train a feature field over the frozen splat: add a low-dimensional feature vector to each Gaussian and optimise it by rasterising features and matching the 2D maps (LangSplat approach: autoencoder compresses 512-d CLIP to a small latent). \[site-aligned\]
2. Quantise the per-Gaussian features into a codebook (k-means, a few hundred entries) so lookups are index-based and fast. \[site: "codebook for rapid lookup"\]

**B3 — Object instances**

1. Cluster Gaussians by (mask ID consistency across views, spatial proximity, feature similarity) into instances; carry the 2D instance IDs from SAM 2 tracking where available \[inferred\].
2. For each instance: name (from GroundingDINO label + VLM confirmation), Gaussian index list, axis-aligned and oriented bounding boxes, view set where it is visible.
3. Foreground/background decision: table-top movable items are foreground; walls, floor, counters, fixed appliances are background. Default rule from object class + VLM confirmation; the user can override by prompt (the page lets you convert "the full scene" or "only foreground objects"). \[site\]

**Agent checkpoint (SCENEAGENT-RUN-STEP) \[site\]:** render each instance's Gaussians alone from 4 views, show the VLM the renders beside the source crop, ask: complete? bleeding into neighbours? wrong label? Re-run with adjusted prompts or merged/split instances on a fail.

**Outputs:** `features.codebook.npz`, `gaussian_feature_idx.npy`, `instances.json` (name, indices, bboxes, fg/bg flag, confidence).

**Software:** GroundingDINO, Segment Anything 2, OpenCLIP, LangSplat (reference implementation), gsplat for feature rasterisation, scikit-learn, Open3D.

## Block C — Background crop and infill

Goal: a background splat with no holes where objects used to sit, plus one clean splat per object. The page states the step ("crop their 3D Gaussians from the background. We then predictively infill these cropped regions") but not the method. \[site\]

&#91;embedded content: Block C flow · 4 steps, 1 retry loop\]

**Measurable outcome (M3):** no visible seams in VLM review at 3 novel views, PSNR drop below 1.5 dB outside the hole, fewer than 50 floater Gaussians per cropped object.

1. **Crop.** Remove each foreground instance's Gaussians from the scene; write them to `objects/<name>/object.ply`, re-centred at the object's bbox centre with the original world transform stored separately.
2. **Detect holes.** Render the cropped background from the training cameras; pixels whose alpha dropped below a threshold relative to the original render define the hole masks per view. \[inferred\]
3. **2D infill.** Inpaint each hole in each training view with a diffusion inpainter (LaMa for speed, Stable Diffusion inpainting for quality), conditioned on the surrounding background. \[inferred — "predictively infill" implies a learned prior\]
4. **3D consistency.** Fine-tune the background splat against the inpainted views for a few hundred iterations, with a mask that only lets new Gaussians grow inside the hole region (Gaussian Grouping / Infusion-style pipeline). \[inferred\]
5. **Object splat clean-up.** Remove floaters: drop Gaussians outside the 2-sigma bbox and those with opacity < 0.05; optionally fine-tune the object splat against its own masked views so the base (the side that touched the table) is not missing. The underside is never observed; Block E's mesh step must close it.

**Agent checkpoint:** VLM compares before/after renders of the background at 3 novel views; fails on visible seams, floating fragments, or missing table surface.

**Outputs:** `background.ply`, `objects/<name>/object.ply`, `objects/<name>/world_T_object.json`.

**Software:** gsplat/Nerfstudio for fine-tuning, LaMa or diffusers inpainting pipeline, Open3D for filtering.

## Block D — Predictive physics and material baking

Goal: each object gets material, mass, friction and a 0–1 rigidity per Gaussian, reduced to object, part and field records, and the simulator gets a physics material it can bind. The page states three sources combined — semantic features, a per-Gaussian predictive prior, and VLM labelling — and section 5 shows the prior's workflow, the heterogeneous-rigidity hat, the marker cap held on by a predicted friction fit (about 2 N breakaway over a 30 mm pull in the page's plot), and "destruction mechanics" such as tearing parchment. \[site\]

&#91;embedded content: Block D flow · the page's section-5 workflow in 7 steps\]

**Measurable outcome (M4, M4b):** median mass error below 30 % on 20 weighed objects, material class accuracy above 85 %, every object settles within 2 s in Isaac Lab, and the marker cap releases in the 1–3 N band the page's plot shows.

**D1 — Per-Gaussian physics prior \[site, section 5\].** The page's own workflow diagram (figure 5.3) fixes this step, so the earlier guesswork is replaced:

1. Masked crops of object k (from Block B masks) → DINOv2 + OpenCLIP → one feature vector per object.
2. That vector is copied to every Gaussian of object k; Gaussians outside any object get the scene-mean vector.
3. For Gaussian i, the input is its object vector plus its own position, size, opacity and colour.
4. A learned physics prior maps that input to a distribution p\_i over material class, mass, friction and a 0–1 rigidity.
5. p\_i is reduced three ways: the object record (mean over k's Gaussians), the part record (inherits the object's record), and a continuous field on the splat (the hat's brim is rigid-red, its crown soft-blue).

The page does not say how the prior itself is trained. \[inferred\]: a small MLP trained on PhysXNet-style labelled assets (material, mass, kinematics), with NeRF2Physics' VLM-grounded material labels as a weak-supervision fallback for classes PhysXNet lacks.

**D2 — VLM estimate \[site\].** Prompt a VLM with 4 object renders, the object name, and the metric bbox; ask for a JSON of {material, mass\_kg, density, static\_friction, dynamic\_friction, restitution, rigidity ∈ \[0,1\], deformable\_type ∈ {rigid, cloth, rope, soft}}, with ranges. Use a structured-output schema.

**D3 — Fusion and bake.** Weighted vote per property: per-Gaussian model for spatial variation (a hat's brim vs crown), VLM for absolute scale (mass), semantic class for defaults. Bake the result into:

- `UsdPhysics:MaterialAPI` on the object prim (static/dynamic friction, restitution) and `UsdPhysics:MassAPI` (mass or density).
- A per-Gaussian property map saved alongside the splat for heterogeneous objects.
- A deformable flag that routes the object to Block E's soft-body path.

**Validation \[inferred\]:** drop test in Isaac Lab (object settles, does not jitter, does not fall through the table); slide test (friction gives plausible stopping distance). Agents read the sim telemetry, not only renders.

**Outputs:** `objects/<name>/physics.json`, `physics_material.usda`, `gaussian_physics.npy`.

**Software:** PyTorch, a VLM with vision input (GPT-4o/Gemini/Claude via API, or Qwen2.5-VL locally for the on-device vision), PhysGaussian or PhysXNet weights, Isaac Lab/PhysX for the drop tests.

## Block E — Mesh extraction, decomposition, articulation, deformables

Goal: a simulatable asset per object: watertight visual mesh, convex collision set, part hierarchy with joints where parts move, and a soft-body/cloth configuration where the object is not rigid. The page shows this working on a radio (door, keys, knobs, selector), a stegosaurus skeleton that walks, a jet engine turbofan, a camera dial and battery door, a Cessna instrument panel, and a range of deformables (towels, hoodie, hose, extension cord, corduroy cap). \[site\]

&#91;embedded content: Block E flow · 4 steps, 1 validation loop\]

**Measurable outcome (M5, M5b):** all user-named movable parts exist on the radio-class object and every joint actuates through its range without interpenetration; the towel drapes and the hose uncoils within 3 s with no NaN velocities.

**E1 — Splat to mesh.** Extract a surface from the object splat (SuGaR, 2DGS, or Gaussian Opacity Fields give cleaner meshes than marching cubes on density). Close the unobserved underside by Poisson reconstruction with a planar prior at the contact face \[inferred\]. Bake the splat colour to a UV texture so the mesh renders in simulators that do not rasterise splats (MuJoCo, Unreal without a plugin).

**E2 — Decomposition \[site: "decompose that object into smaller pieces"\].** Two routes, chosen by the agent:

- *Semantic part split:* reuse Block B's part-level masks (LangSplat's part/subpart levels) to split the mesh into named parts (knob, door, lid). \[site-aligned\]
- *Generative part model:* for objects where the capture does not reveal internal parts (the turbofan blades, the camera's battery compartment), run a part-aware 3D generator such as PartCrafter or HoloPart from the object renders, then register the generated parts to the captured mesh. \[inferred — the page's "articulated and running based on engine schematics" implies a knowledge source beyond the capture\]

Either route also produces convex collision hulls per part (CoACD or V-HACD).

**E3 — Articulation \[site\].** For each movable part the VLM proposes a joint: type (revolute, prismatic, fixed), axis, limits, parent. Geometric candidates come from part bbox symmetry (hinge along the long shared edge, slider along the principal axis). Write `UsdPhysics:RevoluteJoint` / `PrismaticJoint` prims and a drive. Validate by actuating each joint in Isaac Lab and rendering a short clip; the VLM checks the door opens outward, the knob spins in place. The walking stegosaurus "without hand-created animations" implies a generic skeleton-rigging and locomotion routine; treat it as a stretch goal, not core.

**E4 — Deformables \[site\].** Objects flagged `cloth` / `rope` / `soft` in Block D skip rigid decomposition:

- Cloth (towel, hoodie, textile): remesh to a uniform triangle sheet; Isaac Lab/PhysX deformable surface, or a MuJoCo `flex` body. Stiffness and damping from D2 ranges.
- Rope/cable (hose, extension cord): centreline extraction from the mesh skeleton, then a capsule chain with spherical joints or a PhysX rope.
- Soft volume (cap, cardboard): tetrahedralise (fTetWild) and use PhysX FEM soft body.

**Outputs:** `objects/<name>/asset.usd` (visual mesh + collision + joints + materials), `parts/`, `deformable.json`.

**Software:** SuGaR or 2DGS, Open3D, PyMeshLab, CoACD, PartCrafter/HoloPart, fTetWild, pxr (OpenUSD) Python bindings, Isaac Lab for joint validation.

## Block F — Alignment, orientation, USD packaging, inspection

Goal: all assets sit where their real counterparts were, the background splat is oriented to the simulator frame, and one USDZ loads cleanly in Isaac Lab. These are Figure 4 skills 04–07. \[site\]

&#91;embedded content: Block F flow · 4 skills, 1 inspection loop\]

**Measurable outcome (M6, M7, M0):** silhouette IoU above 0.95 per object, table height within 5 mm, USDZ loads with zero PhysX warnings and no object moves more than 2 cm during settle; then the scene-level F1 at 1 cm above 20 % and Chamfer below 3 cm that the page reports.

**F1 — Align (SCENEAGENT-ALIGN-SCENE) \[site\].** Each converted asset's scale, position and rotation is optimised against the original capture by three residuals the page names: photometric (render vs source pixels), silhouette (mask IoU), and edge (Canny/HED edge distance). Initialise from Block C's stored `world_T_object`; optimise 7 DoF (similarity transform) with gradient descent through a differentiable renderer (nvdiffrast or the splat rasteriser). Converge when IoU > 0.95 and photometric loss plateaus.

**F2 — Orient background (SCENEAGENT-ORIENT-BACKGROUND) \[site\].** The page says this is done by "Isaac Lab sweeps": load the scene, sweep candidate rotations/offsets of the background splat, render from the calibrated robot camera, score against the real camera image, keep the best. Also fit the dominant plane (table) and snap it to the simulator ground/ table height; set gravity to -Z.

**F3 — Package (SCENEAGENT-PACKAGE-DATA-PROCESSED) \[site\].** Compose `scene.usda` that references each `asset.usd`, binds physics materials, adds the robot (Franka Panda + UMI gripper USD from the DROID-style setup), places cameras at calibrated poses, records the background splat path, and adds a background collision proxy: a table plane and wall/counter meshes fitted to the splat, with their own friction material, since the splat itself has no colliders \[inferred, required\]. Zip to USDZ. Keep a MuJoCo MJCF export (via a USD→MJCF converter; MuJoCo's own USD importer is still partial) and an Unreal-importable FBX/USD as secondary targets.

**F4 — Inspect (SCENEAGENT-INSPECT-ISAAC) \[site\].** Open the USDZ in Isaac Lab headless; render from 4 camera poses; run a 2-second physics settle; collect: penetration warnings, objects that moved > 2 cm during settle, joint limit violations. The VLM reviews the renders beside the real images and either approves or emits a correction ("mug is 15% too large", "blender rotated 90°"), which loops back to F1. The page names scale, rotation and placement errors as the common failures this catches. \[site\]

**Outputs:** `scene.usdz`, `scene.mjcf` (secondary), `inspection_report.json`, before/after renders.

**Software:** OpenUSD (pxr), Isaac Sim 5.x + Isaac Lab, nvdiffrast, OpenCV, MuJoCo ≥ 3.2 for the MJCF path.

## Block G — Digital sisters (object and scene)

Goal: for each object, 9 sisters that differ in geometry or texture but keep the same affordance; for each scene, many layout + lighting variants exported as separate USDZ files. The page shows 9 sisters per object (flower, Sony radio, teacup) and names generative 3D models as the source. \[site\]

&#91;embedded content: Block G flow · 4 steps, 1 quality gate\]

**Measurable outcome (M8, M9):** 9 of 9 sisters generated per object with CLIP similarity in 0.80–0.95 and mass rescaled by volume; 20 scene layouts per scene with zero interpenetrating starts.

**G1 — Object sisters (SCENEAGENT-OBJECT-SISTERS) \[site\]**

1. Render the processed object from 4 canonical views.
2. VLM writes 9 variation prompts that stay in-class ("same teacup, taller, floral pattern"; never "a bowl").
3. Generate with an image-conditioned 3D model: TRELLIS (image→3D Gaussians/mesh, MIT licence) is the natural fit because it outputs splats; Hunyuan3D-2 or a commercial API (Meshy, Tripo) as alternatives. \[inferred — the page says only "generative 3d models"\]
4. Register each sister to the original (scale to the same bbox height, align principal axes) and copy the parent's physics material, joints and deformable config. Where the sister changes size, rescale mass by volume ratio.
5. Gate: VLM rejects off-class or broken sisters; CLIP similarity to the original must stay above a floor and below a ceiling (too similar is wasted budget). \[inferred\]

**G2 — Scene sisters (SCENEAGENT-SCENE-SISTERS) \[site\]**

1. Sample a layout: for each object slot pick one of its 10 family members, sample a pose inside the task region, drop and settle under physics so every layout is physically reachable. (The page describes this physics-bounded placement in the training section; applying it here keeps scene sisters valid.) \[site\]
2. Randomise lighting: dome-light intensity/colour, 1–2 area lights, shadow softness. \[site: "randomize placement of objects and lighting"\]
3. Export `scene_sister_NNN.usdz` with a manifest of what was sampled, so a failed real-world rollout can be traced to a variant.

**Open question flagged by the page itself:** which factor (lighting, placement, generated geometry) drives the real-world gain is untested; the design should keep each factor switchable so this ablation is cheap later. \[site\]

**Outputs:** `objects/<name>/sisters/01..09/`, `scenes/sister_NNN.usdz`, `sister_manifest.json`.

**Software:** TRELLIS or Hunyuan3D-2, OpenCLIP for similarity gating, OpenUSD, Isaac Lab for the settle step.

## Block H — Agent orchestration layer

Goal: the whole conversion runs from a single natural-language prompt, with an agent per skill that executes a tool, renders the result, inspects it with a VLM, and retries or escalates. The page shows two things that fix the architecture: the user's entry is a prompt ("convert the iPhone capture of this kitchen into a simulation scene with every door, drawer and appliance articulated"), and SceneAgent is packaged as skills that off-the-shelf coding agents load. Section 1.4 benchmarks those agents with and without the skills on the kitchen capture. \[site\]

| Coding agent (model) | F1 at 1 cm, % with skills | F1 at 1 cm, % without | Chamfer, cm with skills | Chamfer, cm without |
| --- | --- | --- | --- | --- |
| Codex (GPT-6 Astra, xhigh) | 41.1 | 16.4 | 3.07 | 4.42 |
| Claude Code (Opus 5.5, extra) | 19.3 | 7.1 | 2.64 | 5.25 |
| ZCode (GLM 5.3, max) | 18.0 | 6.3 | 3.67 | 5.71 |
| Grok Build (Grok 4.7, xhigh) | 17.7 | 6.8 | 2.81 | 5.19 |
| Antigravity (Gemini 3.8 Flash, high) | 20.6 | not completed | 3.05 | not completed |
| Muse (Muse Spark 1.3, xhigh) | 5.1 | not completed | 5.60 | not completed |
| SimFoundry (baseline, no agent) | 10.2 |  | 11.67 |  |

Reading the chart: the skills roughly triple geometric accuracy for every agent, and two agents cannot finish the conversion at all without them. The without-skills videos show flat or empty rooms; the with-skills video shows the full kitchen with articulated cabinets and a mobile robot placed in it. This is the measurable outcome the reproduction must match: a scene-level F1 at 1 cm and Chamfer distance against the capture, computed by a tool the agents never see.

&#91;embedded content: Block H agent loop · 4 stages, retry and hand-off\]

**Measurable outcome (M13, M0):** the same skills complete the kitchen conversion unattended on Codex, Claude Code and one more agent in under 4 h with fewer than 2 human escalations, and the result scores in the page's with-skills band (F1 at 1 cm above 20 %, Chamfer below 3 cm).

**Skill inventory (names from Figure 4) \[site\]:**

| Skill | Owns | Tool it drives |
| --- | --- | --- |
| SCENEAGENT-RUN-PIPELINE | Orchestrator: reads the user's natural-language prompt and the input data, plans which steps are needed, tracks state | Planner only |
| SCENEAGENT-INGEST-DATA | Block A | COLMAP, Nerfstudio |
| SCENEAGENT-RUN-STEP | Blocks B–E, one step at a time with visual review of masks, semantic features, physics properties, background crop and infill | Segmentation, infill, physics, decomposition tools |
| SCENEAGENT-ALIGN-SCENE | F1 | Differentiable alignment |
| SCENEAGENT-ORIENT-BACKGROUND | F2 | Isaac Lab sweeps |
| SCENEAGENT-PACKAGE-DATA-PROCESSED | F3: publishes the `data_processed` package for the editor as USDZ | USD composer |
| SCENEAGENT-INSPECT-ISAAC | F4 | Isaac Lab render + VLM |
| SCENEAGENT-OBJECT-SISTERS | G1 | Generative 3D |
| SCENEAGENT-SCENE-SISTERS | G2 | Layout sampler, USD export |

**Architecture \[inferred\]:**

- *Framework:* one skill directory per row above (SKILL.md plus the tool CLIs it calls), loadable by any coding agent. The page benchmarks Codex, Claude Code, ZCode, Grok Build, Antigravity and Muse with the same skills, so the skills must be agent-agnostic: plain markdown instructions, shell-invocable tools, file-based state. No framework-specific graph. \[site-aligned\]
- *State:* a `pipeline_state.json` per scene (step → status, artifact paths, retry count, VLM verdicts). Every tool is idempotent on its inputs so a retry re-runs only the failed step.
- *Tools:* each block exposes a CLI (`sceneagent ingest --in ... --out ...`) that writes artifacts plus a `report.json`. Agents never edit tensors directly; they call tools and read reports and renders.
- *Visual review contract:* every tool emits a fixed set of renders (`review/view_{0..3}.png`, before/after where relevant). The reviewer prompt is standardised: context, what good looks like, pass/fail, a structured correction if fail.
- *Retry policy:* up to 3 retries per step with the VLM's correction injected as parameters; then escalate to the human via the editor (Block I).
- *Compute routing:* GPU-heavy tools queue on a job runner (Ray or a plain SLURM/Docker queue); the agent loop itself is cheap.

**Edge-device variant \[site, future work\]:** the page imagines the pipeline on a Jetson Orin/Thor with no data leaving the device. That constrains the VLM to a local model (Qwen2.5-VL-7B or similar) and the generative 3D step to a small model; design the tool interfaces so the cloud VLM is a swappable client.

**Software:** Claude Agent SDK or LangGraph, a VLM API, Ray or Docker, Pydantic for report schemas.

## Block I — Interactive editor

Goal: a browser viewer to place robots, add Omniverse assets, crop the background splat and fix conversion errors by hand when the agents stall. The page built this in Three.js. \[site\]

- Rendering: Three.js with a Gaussian-splat renderer (Spark or GaussianSplats3D) for the background and standard mesh loading for assets; USD read via a server-side converter to glTF since browsers do not load USD natively. \[inferred\]
- Edits are stored as a USD overlay (`edits.usda`: transforms, deletions, added references) applied on top of the packaged scene, so the pipeline output is never hand-modified and can be regenerated. \[inferred\]
- Robot placement: load the Panda USD/URDF, drag the base, snap to table height, check reach by drawing the workspace envelope.
- Background crop: box or lasso selection on Gaussians, writes a delete mask back to `background.ply`.
- Asset library: browse NVIDIA Omniverse/Isaac Lab stock assets (kitchen, warehouse, chem lab, data center scenes are all named on the page as hybrid backgrounds). \[site\]

**Software:** Three.js, Spark or GaussianSplats3D, a FastAPI backend wrapping OpenUSD, usd2gltf.

## Block J — Demonstration factory

Goal: thousands of successful, photoreal pick-and-place episodes from the USDZ scene with no human teleoperation. The page describes this block in the most detail of any; nearly all of it is \[site\].

&#91;embedded content: Block J flow · 4 steps, 1 discard loop\]

**Measurable outcome (M10):** 10,000 kept episodes for the lab task (the page's figure), gate pass rate above 70 %, and above 50 composited frames per second per GPU.

**J1 — Initial-state randomisation \[site\]**

- Object pose: uniform over the real task region on the table.
- Object size: scaled within a band (and/or swapped for a sister from Block G).
- Camera and robot base: small perturbations around calibrated poses.
- Physics-bounded: objects are dropped and allowed to settle, so each start state is one the real table could present.

**J2 — Scripted expert \[site\].** Plans from ground-truth object poses in the simulator. Pick-and-place routine: locate → approach → descend → grasp → lift → carry → lower → release → retreat. Implement as waypoints solved with a differential-IK controller in Isaac Lab (or cuRobo for collision-aware motion); grasp pose from the object's oriented bbox top face for the UMI parallel gripper.

**J3 — Success gate \[site\].** After release, check the object is inside the container volume (and at rest); discard failed episodes. Log the fail reason so the physics and layout samplers can be tuned.

**J4 — Observation rendering \[site\].** Each frame is produced by depth-compositing a mesh foreground (robot + objects, rendered by Isaac Sim's RTX path) over the Gaussian-splat background rendered from the same camera. Implementation: render the splat with gsplat at the camera pose to get colour + depth, render the foreground with Isaac Sim with depth, composite per pixel by nearer depth. Before compositing, estimate a dome light from the splat so the RTX foreground matches the capture's lighting, enable shadow catching on the table proxy, and run a colour-matching pass; otherwise the foreground reads as pasted on \[inferred, required\]. Match the real camera intrinsics, resolution and the wrist-camera placement (the evaluation video is a wrist-camera mosaic).

**J5 — Episode recording \[site\].** Camera streams paired with expert actions in a standard format (LeRobot dataset format is the practical choice given pi0.5 and GR00T tooling) \[inferred format\]. Each episode carries a natural-language instruction sampled from many phrasings of the task. \[site\] Target volume: 10,000 episodes for the lab scene, the figure the page quotes. \[site\]

**Throughput note \[inferred\]:** Isaac Lab runs many envs in parallel, but the splat compositing is per-camera per-frame; budget one GPU for \~50–100 fps of composited 224–256 px frames, so 10,000 episodes of \~150 frames each is roughly 4–8 GPU-hours plus simulation time.

**Outputs:** `dataset/` (LeRobot format: parquet episodes + mp4 videos), `episode_stats.json`.

**Software:** Isaac Sim 5.x + Isaac Lab, gsplat, cuRobo (optional), LeRobot.

## Block K — Policy training and real-world evaluation

Goal: a VLA finetuned on simulation episodes only, run zero-shot on the real Franka. The page reports LoRA finetuning of pi0.5 and GR00T 1.6 on 10,000 episodes and real-world success comparable to simulation. \[site\]

&#91;embedded content: Block K flow · 4 steps, 1 training loop\]

**Measurable outcome (M11, M12):** sim success above 70 % over 50 held-out episodes per model; on the real Franka, twin + sisters beats both base and twin-only over 20 trials each, with real within 15 points of sim (the page's "similar performance").

**K1 — Training \[site\].** Model-agnostic imitation learning. Use the openpi (pi0.5) and Isaac-GR00T repositories' LoRA finetuning recipes; inputs are the composited camera streams + instruction, targets are the expert's actions in the robot's native action space (joint or end-effector deltas, must match the real controller). One run per model on 8×H100-class GPUs; keep the LeRobot dataset as the single source so both models train on identical data. \[inferred compute\]

**K2 — Sim evaluation.** Roll out the policy in held-out scene sisters with fresh randomisation; report success over ≥ 50 episodes per task. This gives the sim number the page compares against real.

**K3 — Real hardware \[site\].** Franka Emika Panda, modified DROID setup with the UMI gripper, wrist camera; the evaluation task shown is an eggplant into a basket, recorded as a 3×3 mosaic (real third-person, simulated third-person composited over the splat, wrist camera), with the trial layout defined about a reference point P0 by offsets and yaw angles for both objects; episodes 1, 5 and 18 are illustrated. Reproduce: calibrate camera-to-base, define P0 on the real table, sample the same trial layouts in sim and real, run ≥ 20 trials per condition.

**K4 — Conditions to compare \[inferred, matching the page's claims\].** (a) pretrained VLA, no finetune; (b) finetuned on digital twin only; (c) finetuned on twin + sisters. The page's claim is that (c) improves real-world completion; (b) vs (c) isolates the sister contribution.

**Software:** openpi, Isaac-GR00T, LeRobot, PyTorch, Franka control stack (franka-ros2 or Deoxys/droid-policy-learning), the DROID hardware setup.

## Software and hardware bill of materials

Everything marked **named** appears on the SceneAgent page; everything else is my selection to fill a gap.

| Area | Component | Role | Status |
| --- | --- | --- | --- |
| Capture | iPhone (LiDAR model) / Polycam app | RGB + pose capture | named |
| Capture | COLMAP | SfM poses | named |
| 3DGS | Nerfstudio + gsplat (`splatfacto-big`) | Train and fine-tune splats | named |
| Semantics | GroundingDINO, SAM / SAM 2 | Detection and masks | named |
| Semantics | LangSplat, OpenCLIP | Per-Gaussian language features, codebook | named (LangSplat) |
| Infill | LaMa or diffusers inpainting | 2D hole filling | selection |
| Physics prior | PhysGaussian / PhysXNet / NeRF2Physics | Per-Gaussian material prediction | selection |
| VLM | Claude / GPT-4o / Gemini API; Qwen2.5-VL for on-device | Property estimation, visual review | named (role), selection (model) |
| Meshing | SuGaR or 2DGS, Open3D, PyMeshLab | Splat to mesh, clean-up | selection |
| Decomposition | PartCrafter / HoloPart, CoACD | Parts, convex hulls | selection |
| Deformables | fTetWild, PhysX FEM/cloth, MuJoCo flex | Soft bodies | named (capability) |
| Scene format | OpenUSD (pxr), USDZ | Packaging | named |
| Simulators | Isaac Sim 5.x + Isaac Lab (primary); MuJoCo; Unreal Engine 5 | Physics, rendering, training envs | named |
| Alignment | nvdiffrast or gsplat rasteriser, OpenCV | Photometric/silhouette/edge residuals | named (residuals) |
| Sisters | TRELLIS or Hunyuan3D-2 | Generative 3D variants | named (role) |
| Assets | NVIDIA Omniverse / Isaac Lab stock scenes | Hybrid backgrounds | named |
| Editor | Three.js + Spark/GaussianSplats3D | Browser scene editor | named (Three.js) |
| Agents | Claude Agent SDK or LangGraph | Orchestration, skills | selection |
| Data | LeRobot dataset format | Episode storage | selection |
| Policies | openpi (pi0.5), Isaac-GR00T (GR00T 1.6) | VLA finetuning via LoRA | named |
| Robot | Franka Emika Panda, UMI gripper, DROID-style rig, wrist camera | Real evaluation | named |
| Robot (dex) | Sharpa Wave hands, Dexmate Vega 1 | Dexterous demos (optional) | named |
| Compute | 1 workstation GPU (24 GB+) for conversion; 8× H100/A100 node for VLA finetuning; Jetson Orin/Thor for the edge variant |  | selection |

Licences to check before shipping: Isaac Sim (NVIDIA Omniverse licence), GR00T weights, pi0.5 weights (openpi licence), Hunyuan3D (Tencent community licence); TRELLIS, COLMAP, Nerfstudio, SAM 2 and GroundingDINO are permissive.

## Verified software inventory

Versions and licences checked on 2026-10-10 against the project pages linked; rows without a link are from memory and marked approximate. "Page" = named on the SceneAgent site (including the section-5 and 1.4 content only visible in a browser).

| Tool | Version to pin | Licence | What it does here | Page? | Source |
| --- | --- | --- | --- | --- | --- |
| COLMAP | 3.11 (approximate) | BSD-3 | SfM poses and undistortion for RGB captures | yes |  |
| Nerfstudio | 1.1.x, `splatfacto-big` (needs about 12 GB VRAM) | Apache-2.0 | Train the 3DGS scene; appearance embeddings for colour drift | yes | [nerfstudio releases](https://github.com/nerfstudio-project/nerfstudio/releases) |
| gsplat | 1.x | Apache-2.0 | Rasteriser used by splatfacto; N-D feature and depth rendering for Blocks B and J | implied | [gsplat releases](https://github.com/nerfstudio-project/gsplat/releases) |
| GroundingDINO | Swin-B weights (approximate) | Apache-2.0 | Open-vocabulary detection prompts | yes |  |
| SAM / SAM 2.1 | hiera-large (approximate) | Apache-2.0 | Masks; SAM 2 for cross-frame consistency | yes (SAM) |  |
| DINOv2 | ViT-L/14 (approximate) | Apache-2.0 | Object feature vector for the physics prior | yes |  |
| OpenCLIP | ViT-L/14 (approximate) | MIT | Language-aligned object vector for the physics prior and LangSplat features | yes |  |
| LangSplat | reference repo | approximate | Per-Gaussian language features and codebook | yes (inspiration) |  |
| LaMa / diffusers inpainting | current | Apache-2.0 | 2D infill of cropped regions | no, selection |  |
| PhysXNet / PhysGaussian | reference repos | approximate | Training data and reference code for the physics prior | no, selection |  |
| OpenUSD (pxr) | 25.x (approximate) | Apache-2.0 | Scene composition, UsdPhysics materials, joints, USDZ packaging | yes |  |
| Isaac Sim | 6.1.0 (September 2026); Isaac Lab stable line pairs with 5.1 | NVIDIA proprietary | RTX rendering, PhysX, deformables, inspection renders | yes | [Isaac Sim download](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/download.html) |
| Isaac Lab | 3.0.0-beta2 (July 2026) or 2.3.x stable | BSD-3 | Env framework, demonstration factory, scripted expert | yes | [Isaac Lab container](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/isaac-lab/3.0.0-beta2), [Isaac Lab README](https://github.com/lrz8/IsaacLab) |
| MuJoCo | 3.11.x; USD import experimental | Apache-2.0 | Secondary simulator target | yes | [MuJoCo OpenUSD docs](https://mujoco.readthedocs.io/en/stable/OpenUSD/index.html) |
| mujoco-usd-converter | 0.5.0 | Apache-2.0 | MJCF to USD and MjcPhysics schema | no, selection | [PyPI](https://pypi.org/project/mujoco-usd-converter/) |
| Newton (physics) | 1.5 dev; Isaac Lab 3.0 backend | Apache-2.0 | GPU solver with USD import, MuJoCo Warp backend | no, implied by Isaac Lab 3 | [Newton docs](https://docsearch.algolia.com/mcp/docs/repo/newton-physics/newton) |
| Unreal Engine | 5.x with OpenUSD plugin | Epic EULA | Tertiary target | yes |  |
| TRELLIS / Hunyuan3D-2 | current | MIT / Tencent community | Generative 3D for sisters | no, selection |  |
| CoACD, fTetWild, PyMeshLab, Open3D | current | MIT / MPL-2 / GPL-3 / MIT | Collision hulls, tet meshes, mesh clean-up | no, selection |  |
| Three.js + Spark or GaussianSplats3D | current | MIT | Editor viewer with splat rendering | yes (Three.js) |  |
| openpi | main; `pi05_base` and `pi05_droid` checkpoints; LoRA needs more than 22.5 GB VRAM (RTX 4090 class) | Apache-2.0 | pi0.5 LoRA finetune; reads LeRobot datasets | yes | [openpi README](https://github.com/Physical-Intelligence/openpi) |
| Isaac-GR00T | `n1.6-release` branch; `nvidia/GR00T-N1.6-3B` (about 6 GB); finetune needs about 48 GB VRAM; cameras must be named `wrist` and `front` | NVIDIA model licence | GR00T N1.6 finetune; reads LeRobot v2 plus `modality.json` | yes | [NVIDIA playbook](https://build.nvidia.com/station/gr00t/instructions), [Seeed wiki](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/) |
| LeRobot dataset format | v2.1 (parquet + mp4) | Apache-2.0 | Common episode format both VLAs accept | no, selection |  |
| Coding agents | Codex, Claude Code, ZCode, Grok Build, Antigravity, Muse | per vendor | Hosts for the SCENEAGENT-\* skills | yes (section 1.4) |  |
| SimFoundry | baseline in the chart | unknown | Non-agent baseline the page compares against | yes |  |

Two compatibility traps from the checks: Isaac Lab's stable 2.3 line is validated against Isaac Sim 4.5–5.1, not the 6.x downloads, so pin the pair the Isaac Lab README lists; and `pi05_droid` is the better starting checkpoint for a DROID-style Franka rig than `pi05_base`, because the camera layout and action space already match.

## Data contracts between blocks

Every block reads and writes the same scene workspace. Fixing these paths and schemas early is what lets agents retry single steps. All of this is \[inferred\] implementation design.

```
scene_<id>/
  input/                 raw frames | capture.ply | mesh/ | lidar/
  ingest/                scene.ply, cameras.json, scale.json, ingest_report.json
  semantics/             features.codebook.npz, gaussian_feature_idx.npy, instances.json
  background/            background.ply, holes/, review/
  objects/<name>/
    object.ply           cropped splat (object frame)
    world_T_object.json  original pose in scene frame
    physics.json         fused properties + provenance per property
    asset.usd            mesh + collision + joints + bound materials
    parts/, deformable.json
    sisters/01..09/      same layout as the parent
    review/              fixed render set for VLM checks
  scene/                 scene.usda, scene.usdz, scene.mjcf, inspection_report.json, edits.usda
  sisters/               sister_NNN.usdz, sister_manifest.json
  dataset/               LeRobot episodes
  pipeline_state.json    step status, retries, verdicts
```

Conventions: metres, Z-up, right-handed, quaternions as (w, x, y, z); every `*.json` validated by a Pydantic schema; every tool writes `report.json` with `{status, metrics, renders[], suggested_fixes[]}` so the agent's reviewer prompt is identical across steps.

## Work breakdown and phasing

Five phases, each ending in a demo that the next phase builds on. Durations are a working estimate for a team of 4 (2 perception/3D, 1 simulation/USD, 1 robot-learning) \[inferred\].

| Phase | Weeks | Deliverable | Blocks |
| --- | --- | --- | --- |
| 0 Setup | 1–2 | Workspace layout, CLIs scaffolded, Isaac Lab + Panda USD loading, capture rig calibrated | H skeleton, K3 hardware |
| 1 Twin v0 | 3–6 | One tabletop scene from iPhone to USDZ with rigid objects, manual physics values, manual alignment; loads in Isaac Lab | A, B, C, E1, F3 |
| 2 Predictive + agentic | 7–11 | Physics prediction, alignment and inspection loops run unattended; 3 retries then human escalation; editor usable | D, F1, F2, F4, H, I |
| 3 Demo factory + first policy | 12–15 | 10,000 episodes on the twin; pi0.5 LoRA finetune; sim and real success numbers | J, K |
| 4 Sisters + articulation + deformables | 16–22 | 9 sisters per object, scene sisters, articulated radio-class object, one cloth and one rope object; ablation (twin vs twin+sisters) | G, E2–E4, K4 |
| 5 Breadth | 23+ | Photogrammetry/LiDAR/generated-splat inputs, MuJoCo and Unreal exports, dexterous-hand embodiment, Jetson variant | A2, F3 secondary, H edge |

Gate for each phase: the previous phase's artifact is regenerated from scratch by the pipeline with no manual edits (the `edits.usda` overlay must be empty).

## Code milestones with measurable outcomes

Each milestone is a command you can run and a number it must produce. The two scene-level numbers at the top (F1 at 1 cm and Chamfer distance against the capture) are the page's own benchmark metrics and are the project's headline; everything below them is a per-block gate. Targets marked \[site\] come from the page; the rest are engineering targets.

| # | Milestone (what runs) | Metric | Target | How it is measured |
| --- | --- | --- | --- | --- |
| M0 | `sceneagent eval-scene scene.usdz capture/` on any scene | F1 at 1 cm (%), Chamfer distance (cm) | F1 above 20 % and Chamfer below 3 cm for the kitchen capture, i.e. the Claude Code / Codex with-skills band \[site\] | Sample 1 M points from the converted meshes and from the capture splat (or photogrammetry mesh); compute symmetric Chamfer and F1 with 1 cm threshold, in metres after Block A scale |
| M1 | `sceneagent ingest` on the iPhone kitchen video | PSNR / LPIPS on 10 % held-out views; metric scale error | PSNR above 27 dB, LPIPS below 0.20; scale error below 2 % against the tag board | Nerfstudio `ns-eval`; measured tag size vs known size |
| M2 | `sceneagent run-step segment` | Instance mask mIoU vs 20 hand-labelled views; instance count error | mIoU above 0.80; count within ±1 of ground truth per scene | Hand labels in Label Studio; script compares rendered instance masks |
| M3 | `sceneagent run-step crop-infill` | Background PSNR at 3 novel views after infill; floater count | No visible seams in VLM review; PSNR drop below 1.5 dB vs uncropped background outside hole; fewer than 50 floater Gaussians per object | Render before/after; count Gaussians outside 2σ bbox |
| M4 | `sceneagent run-step physics` on 20 objects with known mass | Mass relative error; material class accuracy; drop-test stability | Median mass error below 30 %; class accuracy above 85 %; 100 % of objects settle within 2 s with motion below 1 cm | Kitchen scale for ground truth; Isaac Lab drop test telemetry |
| M4b | Marker cap friction fit | Breakaway force from pull test | Pull force rises then releases, in the 1–3 N band the page's plot shows \[site\] | Isaac Lab prismatic joint pull with force sensor |
| M5 | `sceneagent run-step decompose` + `articulate` on the radio-class object | Parts found; joints validated | All user-named movable parts exist; every joint actuates through its range without interpenetration | Automated joint sweep in Isaac Lab; VLM confirms motion direction |
| M5b | Deformable objects (towel, hose) | Settles under gravity; no explosion | Cloth drapes within 3 s with no NaN velocities; rope uncoils | PhysX telemetry |
| M6 | `sceneagent align-scene` + `orient-background` | Silhouette IoU; photometric residual; table-plane height error | IoU above 0.95 per object; table height within 5 mm of calibrated real table | Differentiable render vs capture views |
| M7 | `sceneagent package` + `inspect-isaac` | USDZ loads; settle displacement; inspector verdict | Loads with zero PhysX warnings; no object moves more than 2 cm during settle; VLM approves within 3 retries on 80 % of scenes | Isaac Lab headless script; `inspection_report.json` |
| M8 | `sceneagent object-sisters` on 3 objects | 9 sisters each; CLIP similarity to original; physics copied | 9 of 9 generated; similarity in 0.80–0.95; mass rescaled by volume | OpenCLIP cosine; manifest check |
| M9 | `sceneagent scene-sisters` | Variants exported; all physically settled | 20 layout variants per scene, zero interpenetrating starts | Settle test per variant |
| M10 | Demonstration factory, 1 task | Kept episodes; success-gate rate; frames per second | 10,000 kept episodes \[site\]; gate pass rate above 70 %; above 50 composited fps per GPU | `episode_stats.json` |
| M11 | pi0.5 and GR00T N1.6 LoRA finetune | Training loss converges; sim success on held-out sisters | Sim success above 70 % over 50 episodes per model | Isaac Lab rollout script |
| M12 | Real-robot evaluation, eggplant into basket | Real success rate for (a) base, (b) twin only, (c) twin + sisters | (c) higher than (a) and (b) over 20 trials each; real within 15 points of sim \[site: "similar performance"\] | Trial sheet with P0 layouts; video record |
| M13 | Full pipeline from one prompt, 3 agents | Unattended completion; wall-clock; human escalations | Completes on Codex, Claude Code and one more agent; under 4 h per tabletop scene; fewer than 2 escalations | `pipeline_state.json` |

Run M0 after every milestone from M6 onward; it is the number the page reports and the one a reviewer will ask for first.

## Critical self-review of the draft

I re-read the draft against the page as a sceptical reviewer. Findings, with what I changed:

**Steps the first draft missed (now added or flagged)**

1. *Background collision geometry.* A splat has no colliders, so objects would fall through the table. Added to F3: fit a table plane and wall proxies with their own friction material. The page never mentions this, but it is unavoidable.
2. *Metric scale.* COLMAP is scale-free; mass, density and robot reach all need metres. Added to A1 step 5. Not on the page.
3. *Lighting and shadow consistency in the composite.* J4 composites an RTX-rendered foreground over a splat background. Without an estimated environment light from the capture and a shadow-catcher on the table proxy, the foreground will look pasted on and the policy may learn that cue. Not on the page; add to J4 as a required sub-step: estimate an HDRI-like dome light from the splat, enable shadow catching on the table proxy, and run a colour-matching pass.
4. *Object undersides.* Never observed in a capture; without closure the mesh is open and the collision hull is wrong. Added to C5 and E1.
5. *Ablation hooks.* The page itself says the lighting/placement/geometry contribution is untested; G2 keeps each factor switchable so the ablation is free.

**Places where I could have over-claimed (now labelled \[inferred\])**

- The infill method (diffusion inpainting + masked splat fine-tune), the per-Gaussian physics model identity (PhysGaussian/PhysXNet/NeRF2Physics), the generative 3D model for sisters (TRELLIS/Hunyuan3D), the decomposition model (PartCrafter/HoloPart), the dataset format (LeRobot), the agent framework, and all compute/time estimates are my choices. The page names the roles, not the tools.
- The page says SAM; I recommend SAM 2 for video-consistent masks. That is a judgement, not a report.
- "Walking stegosaurus without hand-created animations" and "engine running based on schematics" imply capabilities (auto-rigging, external knowledge injection) that I have not designed in detail; they are out of the core scope.

**Corrections made in this pass**

- Isaac Sim version bumped from 4.x to 5.x (the current major line); TRELLIS licence corrected to MIT.
- Scene-sister placement now explicitly reuses the drop-and-settle rule the page describes for training randomisation, rather than inventing a separate sampler.

**Things that still look logically fragile**

- *USDZ as the only interchange.* MuJoCo's USD import is partial and Unreal's needs the OpenUSD plugin; the MJCF/FBX secondary exports in F3 are a hedge, not a solved path. Expect per-simulator physics parameter mismatches (PhysX vs MuJoCo friction models).
- *Splat rendering inside Isaac.* Isaac Sim's RTX renderer does not rasterise Gaussians, which is exactly why the page depth-composites. Any design that assumes the splat is visible from inside the simulator's own viewport is wrong; the editor (Block I) is where splat and meshes are seen together.
- *VLM review as the only QA.* Agents can approve a plausible-looking but metrically wrong scene. D's drop/slide tests and F4's settle telemetry are there so the loop has numeric checks, not only renders. Keep those.
- *Real-world numbers.* The page says results are still being evaluated; the design should not promise a success-rate target until K4's ablation is run.

**What the browser pass changed (2026-10-10).** The first draft was written from a text fetch that missed JavaScript-rendered sections. Opening the page in a browser corrected four things: the skill names are SCENEAGENT-*, not REAL2SIM-*; the pipeline entry is a natural-language prompt; SceneAgent is packaged as skills for coding agents and benchmarked with and without them (F1 at 1 cm and Chamfer distance, now the project's headline metrics in the milestone table); and section 5 gives the physics prior's actual workflow (DINOv2 + OpenCLIP object vector copied to each Gaussian, plus per-Gaussian geometry, through a learned prior to a per-Gaussian distribution), which replaced my earlier guess in Block D. Still unverified: the paper and code (unreleased), how the prior is trained, and the exact alignment-residual weights. Videos were sampled at a few frames each, not watched end to end.

Build sequence, tests, change log and the laptop/Vast/OpenAI topology for your repo: Build plan for your repo
