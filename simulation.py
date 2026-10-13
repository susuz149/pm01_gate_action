import mujoco
import mujoco.viewer
from pathlib import Path

#SCENE_PATH = Path(__file__).parent / "robot/pm01.xml"
SCENE_PATH = Path(__file__).parent / "assets/resource/pm_v2.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)