# SceneAgentX research handoff: SimFoundry vs Harvard vs our masking/mesh issues

## Meta

| Field | Value |
|---|---|
| Date context | 2026-10-29 |
| Repo | `c:\Users\sande\cursor\SceneagentX` |
| Branch typically | `cursor/add-polycam` |
| Harvard paper PDF (user path) | `C:\Users\sande\Downloads\harvard team arxive apper.pdf` (arXiv:2610.34745 SceneAgent) |
| SimFoundry | https://github.com/NVlabs/SimFoundry , arXiv:2606.28276 |
| Dual-doc plan | `.cursor/plans/sceneagent_dual_docs_68a7cda8.plan.md` |
| Stance | **SimFoundry is a compared system, NOT a method to copy wholesale** |

Read this file before changing mask membership, crop, Block B, or mesh sizing. Paper wins over website guesses. A skip is never a pass.

---

## 1 — Core representation difference

| System | Pipeline shape |
|---|---|
| **SimFoundry (SF)** | crop → generate mesh (Hunyuan / TRELLIS) → place in sim. Object Gaussians: basically **no** (background splat optional). |
| **Us / Harvard SceneAgent** | splat → select Gaussians via masks → crop / infill → mesh / physics from the **selected** field. |

Why a room-sized mask becomes a room-sized mesh on our path but not on SF:

1. We keep (and later mesh) the Gaussians that survive membership. If the 2D mask covers the room, membership pulls in room Gaussians; crop and Poisson/mesh inherit that extent.
2. SF never builds the object mesh from selected scene Gaussians. It masks an object out of RGB/depth for removal ordering and background, then **generates** a fresh object mesh from the crop. A bad room-wide mask hurts detection/inpaint order; it does not inflate an object mesh built from a room-sized Gaussian set.

Fixing our room-mesh failure is a **membership / prompt / gate** problem on the Harvard path—not a reason to switch the object carrier to SF generative meshes.

---

## 2 — Harvard paper method (§§3.2–3.3)

Facts from arXiv:2610.34745 (SceneAgent). Dual-doc plan: `.cursor/plans/sceneagent_dual_docs_68a7cda8.plan.md`.

### Scene and object state

- Scene starts as **3DGS**.
- Object \(k\) is ok as \((P_k\) Gaussians, \(M_k\) mesh, \(T_k\) pose, \(\theta_k\) physics, \(J_k\) joints\).

### Masks

- SAM on views; default **every 10th** view.
- S2 = grounded masks + visual QA.

### Membership (equation 1)

A Gaussian is kept only if it:

1. **projects inside the mask**, and
2. lies **near the front surface** (absolute slack \(\varepsilon_a = 0.08\), relative slack \(\varepsilon_r = 0.03\)).

### OpenCLIP / DINOv2

Feed the **physics prior ONLY** — they do **not** decide membership.

### Multi-view merge (SplatGraph)

- Centroid / containment / Jaccard thresholds (paper); experiments use \(\kappa = 1\) → **union** kept.

### Mesh

- **SAM 3D**, then **trimmed similarity ICP** seeded to align to \(P_k\).

### Infill

- Plane completion; optional **LaMa**.

### Agents

Inspect / repair in the simulator (agent loop). This is a claimed SceneAgent gap vs SimFoundry Table 1.

---

## 3 — SimFoundry method

Compared system only. Do not wholesale-copy generative meshes into SceneAgentX object path.

### Stages (high level)

- Stages **1–14** under **A_reconstruction**
- **B** cousins
- **C** OG smoke

### Reconstruction loop (from one RGB video)

1. **VLM** names objects.
2. **SAM3** on keyframe; **SAM2** track; **iterative remove-one** + RGB/depth inpaint; order by occlusion / support / depth.
3. **Ground plane** separate (s3 SAM3 + RANSAC).
4. Metric depth: **DA3** / **FoundationStereo**.
5. Mesh: **Hunyuan2.1** / **TRELLIS.2**; pose: **FoundationPose**.
6. **Articulate Anything** + critic VLM.
7. **CoACD** + **PyBullet** settle.
8. Auto background: **SAM2** track + dilate + **VOID** 2-pass + depth-supervised splat.

### Gates (SF style — different from SceneAgentX)

SF uses operational gates such as `StageResult` / `stage_info.json`, frame-selection hard gates, mesh `SystemExit` on failure, articulation URDF checks, physics settle.

These are **NOT** SceneAgentX pytest milestone gates. SF skip ≠ SceneAgent skip≠pass discipline (changelog: a skipped check is never a pass; nothing is retrained to make a number move).

---

## 4 — Comparison matrix

| Topic | Harvard | SimFoundry | Our code |
|---|---|---|---|
| Scene carrier | 3DGS first; objects from selected Gaussians + mesh | RGB video → remove-one; gen meshes; optional bg splat | 3DGS / splatfacto first; object from mask lift |
| Masks | SAM every 10th view; S2 grounded + QA | VLM names; SAM3 keyframe; SAM2 track; remove-one | GroundingDINO caption + SAM 2.1 per frame; no video track |
| Object Gaussians? | Yes — \(P_k\) is core | Basically no (bg splat optional) | Yes — lift / crop / field from selected Gaussians |
| Membership | Eq.1: in-mask **and** front surface (\(\varepsilon_a=0.08\), \(\varepsilon_r=0.03\)) | Mask for crop/remove/inpaint, not Gaussian membership | Mask projection + **eq.1 front-surface keep in `object_crop_iou`** (\(\varepsilon_a=0.08\), \(\varepsilon_r=0.03\)) |
| Multi-view merge | SplatGraph thresholds; \(\kappa=1\) union | Per-object remove-one loop | No SplatGraph thresholds |
| CLIP / DINO role | Physics prior **only** | VLM / critic / naming roles (not Harvard membership) | LangSplat / physics vector; risk of treating field as membership |
| Mesh | SAM 3D + trimmed similarity ICP to \(P_k\) | Hunyuan2.1 / TRELLIS.2 + FoundationPose | Poisson / 2DGS-style from splat; Chamfer while metres null |
| Background / holes | Plane completion; optional LaMa | VOID inpaint + depth-supervised bg splat | LaMa coded; often not run; plane completion absent |
| Room-sized mask | Tight membership + agent repair | Gen mesh from object crop — room mask does not become room Gaussian mesh | Room mask → room Gaussians → room mesh (failure mode) |
| Repair loop | Agents inspect/repair in sim | SF paper path; SceneAgent Table 1 claims SF lacks agent repair | Block H / iterate; dual-doc skip≠pass |

---

## 5 — Our concrete failure (can orbit2)

1. **Orbit2 mask covered the room.**
2. `configs/can_block_b_prompt.txt` was a **synonym list**, not a user story (historically e.g. aluminum soda can / cylindrical beverage can / drink can with pull tab — not one of the five dual-doc stories).
3. **Dual-doc rule:** room job + room mask → **one** tighter `box_threshold` retry → refuse; second miss stops; **no splat retrain**.
4. That rule **did not own the run historically** (synonym caption / no `request.json` story gate).
5. Related **bottle** issue: wide mesh vs tight crop (`fuse_bottle_bounds` / `alpha_fuse`) — same family: **selection too big → mesh too big**.

---

## 6 — Dual-doc prompt stories

From `.cursor/plans/sceneagent_dual_docs_68a7cda8.plan.md`. The user's sentence chooses the job. A second miss stays refused. A skip is not a pass.

| Story | Example prompt | Mask behavior | When infill schedules |
|---|---|---|---|
| Close-up / orbit, one object | "This is an orbit video of the can. I want the can." | Detect that name. A large mask does **not** refuse the job. | Not scheduled |
| Room, one named object | "Here is the room. I want the can." | Detect only the can. If mask covers the room: **one** tighter try; second miss **refused**. | Only if prompt asks for the room with the can removed |
| Several named objects | "I want the can and the bottle." | One mask and one object file per name. Boxes not merged. | Only for an object the prompt says to remove |
| Unnamed foreground | "Only the foreground objects." | VLM names them; each name gets its own mask. | Only for objects the prompt says to remove |
| Full scene | "Turn this kitchen into a simulation with every door, drawer, and appliance." | Walls stay. Only objects to remove are cropped. | Only those removed objects |

Harvard page examples (27 Oct 2026): lab clamp jaw/screw; kitchen every door/drawer/appliance.

---

## 7 — Best fix (implemented / wired)

Work from the **[Fix room-mask membership gates]** agent. Wired locally; do not assume Vast can orbit2 already re-ran under the room story.

### Behavior

| Piece | What it does |
|---|---|
| `request.json` story required for can Block B | `vast-cluster/scripts/launch_can_block_b_rerun.py` refuses synonym-only / missing job+target+prompt |
| `mask_downstream_block_reason()` | Blocks crop / field / B4 while `mask_retry` **pending** or `target_mask_check` **stop** |
| One `box_threshold` retry | Default **0.25 → 0.30** (`raised_box_threshold`); then refuse |
| Host marker | `STOP_AFTER_FIELD_REFUSE` — stop before B4 after room-mask field refusal |
| `agent_loop` | Prefers `request.json`; refuses pending room-mask retry |
| `crop_decision` (`block_c.py`) | Refuses while pending / stopped |
| Missing overlay | `asked: false` (not a pass) |
| `front_surface_keep` / `front_surface_keep_mask` | Harvard eq.1 (\(\varepsilon_a=0.08\), \(\varepsilon_r=0.03\)) — **wired into `object_crop_iou` lift** |

### Tests (reported)

`61 passed`, `2 skipped` on:

- `tests/test_task_spec.py`
- `tests/test_block_02_see.py`
- `tests/test_block_c.py`
- `tests/test_block_b_job.py`
- `tests/test_block_h_prompt.py`

### Files touched

- `tools/block_b_job.py`
- `tools/block_c.py`
- `tools/task_spec.py`
- `tools/agent_loop.py`
- `vast-cluster/scripts/launch_can_block_b_rerun.py`
- `vast-cluster/scripts/object_langsplat_b2.py`
- `vast-cluster/scripts/object_langsplat_b3.py`
- matching tests
- `CHANGELOG.md` (`[Unreleased]` bullets naming those files)

---

## 8 — Remaining gaps

1. **Live can orbit2 Vast rerun** with a real room story (`request.json`) **not done** (eq.1 is wired locally; Vast not re-verified).
2. **PR split 1–3 paused**:
   - Local branch `pr/polycam-mesh-ingest` exists (1 commit, **not** pushed as PR).
   - Backup ref: `refs/backup/pre-split-20261029192617`.
   - Working on `cursor/add-polycam` with **dirty tree restored**.

---

## 9 — Key code pointers

| Path | Role |
|---|---|
| `c:\Users\sande\cursor\SceneagentX\tools\task_spec.py` | Store prompt; room-mask one retry; gate overlay; `asked:false` |
| `c:\Users\sande\cursor\SceneagentX\tools\pipeline.py` | `apply_mask_retry` — detect+lift once; no splat retrain |
| `c:\Users\sande\cursor\SceneagentX\tools\block_b_job.py` | `mask_downstream_block_reason`, `front_surface_keep` / `front_surface_keep_mask` (eq.1), host B1–B4 |
| `c:\Users\sande\cursor\SceneagentX\tools\block_c.py` | `crop_decision` refuses pending/stopped |
| `c:\Users\sande\cursor\SceneagentX\vast-cluster\scripts\object_semantics.py` | Detection / overlays |
| `c:\Users\sande\cursor\SceneagentX\vast-cluster\scripts\object_crop_iou.py` | Lift / membership projection with Harvard eq.1 front-surface keep |
| `c:\Users\sande\cursor\SceneagentX\vast-cluster\scripts\object_langsplat_b2.py` | B2 |
| `c:\Users\sande\cursor\SceneagentX\vast-cluster\scripts\object_langsplat_b3.py` | B3 |
| `c:\Users\sande\cursor\SceneagentX\vast-cluster\scripts\launch_can_block_b_rerun.py` | Can Block B requires `data/can/request.json` story |
| `c:\Users\sande\cursor\SceneagentX\.cursor\plans\sceneagent_dual_docs_68a7cda8.plan.md` | Dual-doc + paper correction |
| `c:\Users\sande\cursor\SceneagentX\configs\can_block_b_prompt.txt` | Synonym list — **not** a dual-doc user story |
| `c:\Users\sande\cursor\SceneagentX\.cursor\rules\changelog.mdc` | Skip ≠ pass; Unreleased bullets for tools/tests |

Related bottle sizing: `vast-cluster/scripts/fuse_bottle_bounds.py`, `vast-cluster/scripts/alpha_fuse_bottle.py`.

---

## 10 — Verify commands for can

From repo root `c:\Users\sande\cursor\SceneagentX`:

```bash
python -c "from tools.task_spec import accept_pipeline_prompt; raise SystemExit(accept_pipeline_prompt('can', 'Here is the room. I want the can.'))"
python -c "from tools.task_spec import gate_detected_target; from tools.pipeline import apply_mask_retry; c=gate_detected_target('can'); raise SystemExit(c if c else apply_mask_retry('can'))"
python vast-cluster/scripts/launch_can_block_b_rerun.py start
```

**Expect:**

- Pending / stop blocks field / B4 / crop.
- Second room lift refuses.
- No splat retrain.

Before any Vast SSH: `vastai show instances` and use the **running** instance (see `.cursor/rules/vast-ssh.mdc`). Key: `%USERPROFILE%\.ssh\ssh_sceneagentX_1`, user `root`.

---

## 11 — Prompt template for the next agent

### Default (copy-paste)

```
Read this handoff end-to-end before editing code:

1) C:\Users\sande\AppData\Local\Cursor\AgentStores\cursor_agent_stores\42966ee9-8be0-42a9-92f5-60184eb66a76\files\handoffs\simfoundry-harvard-mask-mesh-handoff.md
2) c:\Users\sande\cursor\SceneagentX\docs\handoffs\simfoundry-harvard-mask-mesh-handoff.md
(Same content; prefer whichever is available.)

Also read: .cursor/plans/sceneagent_dual_docs_68a7cda8.plan.md (prompt stories + paper correction)
and .cursor/rules/changelog.mdc.

Goal (user will choose; if unclear, ASK once then default to A):
  A) Wire Harvard eq.1 front-surface keep into the lift (object_crop_iou / membership path) with tests.
     Use tools/block_b_job.py:front_surface_keep (εa=0.08, εr=0.03). Do NOT copy SimFoundry generative meshes (Hunyuan/TRELLIS) as the object carrier.
  B) Continue masking/membership hardening (gates, request.json story, room refuse) without marking skips as pass.
  C) Run can orbit2 verify on Vast with a real room story in data/can/request.json.

Defaults if user does not choose:
  (A) wire eq.1 into lift + tests
  (B) do not mark skips as pass
  (C) do not commit unless asked

Constraints:
- SimFoundry is compared system only — not a wholesale method to copy.
- Dual-doc: room job + room mask → one box_threshold 0.25→0.30 retry → refuse; second miss stops; no splat retrain.
- Changelog: every tools/ or tests/ change gets an [Unreleased] bullet naming the file; skip ≠ pass; never edit closed block sections.
- Do not rent a second Vast machine unless asked; refresh vastai show instances before SSH.
- Do not commit or push unless the user explicitly asks.

Success criteria:
- Eq.1 (if A): front-surface filter applied in lift with unit tests; far-behind-mask Gaussians dropped; room-sized mesh failure mode reduced without SF gen-mesh switch.
- Gates (if B): pending mask_retry / target_mask_check stop still block crop/field/B4; missing overlay asked:false.
- Vast (if C): can orbit2 rerun under "Here is the room. I want the can." (or equivalent); second room lift refuses; no splat retrain; evidence logged.
- Tests green for touched modules; CHANGELOG [Unreleased] updated.
```

### Alternate — Vast can orbit2 room mask only

```
Handoff: c:\Users\sande\cursor\SceneagentX\docs\handoffs\simfoundry-harvard-mask-mesh-handoff.md
(also agent store copy under handoffs/simfoundry-harvard-mask-mesh-handoff.md)

Task: debug LIVE can orbit2 room mask on Vast ONLY.
1. vastai show instances — use running host; key %USERPROFILE%\.ssh\ssh_sceneagentX_1 as root.
2. Ensure data/can/request.json is a room story (job=room, target=can, prompt like "Here is the room. I want the can.") — not configs/can_block_b_prompt.txt synonyms.
3. python vast-cluster/scripts/launch_can_block_b_rerun.py start (or equivalent host job).
4. Confirm: one box_threshold 0.25→0.30 retry; pending/stop blocks field/B4/crop; second room lift refuses; no splat retrain.
5. Do not wire eq.1 in this pass unless lift evidence proves depth is available and needed.
6. Do not commit unless asked. Skip ≠ pass.
```

---

Do not treat SimFoundry 6.9% F1 as their published score; different test from SceneAgent bin protocol.
