import mujoco
from pathlib import Path

# 设置路径
urdf_path = "robot/pm01.urdf"
output_path = "robot/pm01.xml"

# 1. 加载 URDF
# MuJoCo 会在加载时自动在内存中将其转换为 MJCF 结构
model = mujoco.MjModel.from_xml_path(urdf_path)

# 2. 将内存中的模型保存为 MJCF (XML) 格式
# 这会生成一个包含所有 Body 嵌套、惯量和关节定义的标准 MuJoCo XML
mujoco.mj_saveLastXML(output_path, model)

print(f"转换成功！文件已保存至: {output_path}")