"""One Isaac RGB attempt for the existing scan USD. No retry loop."""

from pathlib import Path

USD = "/root/SceneagentX/data/scan/inspect/usd/scan.usd"
OUT = Path("/root/SceneagentX/data/scan/inspect/isaac_rgb.png")

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
simulation_app = launcher.app

import isaaclab.sim as sim_utils
import omni.replicator.core as rep
import omni.usd
from isaaclab.sim import SimulationContext

context = omni.usd.get_context()
context.open_stage(USD)
simulation = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0))
simulation.reset()
camera = rep.create.camera(position=(0.4, -0.4, 0.3), look_at=(0.0, 0.0, 0.05))
product = rep.create.render_product(camera, (640, 480))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=str(OUT.parent), rgb=True)
writer.attach([product])
rep.orchestrator.step()
frames = sorted(OUT.parent.glob("rgb_*.png"))
if not frames:
    raise SystemExit("reset returned but no rgb png was written")
frames[-1].replace(OUT)
print("ISAAC_FRAME_WROTE", OUT)
simulation_app.close()
