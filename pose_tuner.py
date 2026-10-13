"""
PM01 姿态调试工具
用键盘调整双臂关节角度，找出各阶段合适的目标姿态。
下半身完全固定，只控制双臂。

操作说明：
  数字键 1-5：选择左臂关节（1=肩pitch, 2=肩roll, 3=肩yaw, 4=肘pitch, 5=肘yaw）
  数字键 6-0：选择右臂关节（6=肩pitch, 7=肩roll, 8=肩yaw, 9=肘pitch, 0=肘yaw）
  Q / A：当前关节角度 +0.1 / -0.1 弧度（大调整）
  W / S：当前关节角度 +0.05 / -0.05 弧度（中调整）
  E / D：当前关节角度 +0.01 / -0.01 弧度（微调整）
  R：重置到初始姿态
  P：打印当前左右臂角度（方便复制到主程序）
  Esc：退出
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "assets/resource/pm_v2.xml"

LEFT_JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
]

RIGHT_JOINT_NAMES = [
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
]

KP = np.array([100.0, 100.0, 80.0, 80.0, 50.0])
KD = np.array([4.0, 4.0, 3.0, 3.0, 2.0])


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 查找双臂关节 ID
    left_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_JOINT_NAMES]
    right_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_JOINT_NAMES]

    # 查找双臂执行器 ID
    def find_actuator(jid):
        for aid in range(model.nu):
            if model.actuator_trnid[aid, 0] == jid:
                return aid
        return -1

    left_act_ids = [find_actuator(j) for j in left_joint_ids]
    right_act_ids = [find_actuator(j) for j in right_joint_ids]

    # 关节限位
    left_limits = [model.jnt_range[jid] for jid in left_joint_ids]
    right_limits = [model.jnt_range[jid] for jid in right_joint_ids]

    # 打印信息
    print("左臂关节:")
    for i, n in enumerate(LEFT_JOINT_NAMES):
        print(f"  {n}  id={left_joint_ids[i]}  限位=[{left_limits[i][0]:.3f}, {left_limits[i][1]:.3f}]  执行器={left_act_ids[i]}")
    print("右臂关节:")
    for i, n in enumerate(RIGHT_JOINT_NAMES):
        print(f"  {n}  id={right_joint_ids[i]}  限位=[{right_limits[i][0]:.3f}, {right_limits[i][1]:.3f}]  执行器={right_act_ids[i]}")

    # 目标角度（初始自然下垂姿态）
    q_left_target = np.array([0.05, 0.15, -0.1, -0.2, 0.0])
    q_right_target = np.array([0.05, -0.15, 0.1, -0.2, 0.0])

    # 当前选中的关节
    selected_arm = "left"
    selected_idx = 0

    # --- 记录下半身所有关节的初始角度，用来硬固定 ---
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.82
    data.qpos[3] = 1.0

    # 先设置双臂初始角度
    for i, jid in enumerate(left_joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = q_left_target[i]
    for i, jid in enumerate(right_joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = q_right_target[i]

    # 记录所有非双臂关节的初始 qpos 和 qvel 索引，用来硬固定
    arm_jid_set = set(left_joint_ids + right_joint_ids)
    body_joints = []  # 需要硬固定的关节
    for jid in range(model.njnt):
        if jid not in arm_jid_set:
            qadr = model.jnt_qposadr[jid]
            vadr = model.jnt_dofadr[jid]
            q0 = data.qpos[qadr]
            body_joints.append((jid, qadr, vadr, q0))

    print(f"\n总关节数: {model.njnt}, 双臂关节: {len(arm_jid_set)}, 其他关节(硬固定): {len(body_joints)}")

    mujoco.mj_forward(model, data)

    def apply_delta(delta):
        nonlocal q_left_target, q_right_target
        if selected_arm == "left":
            q_left_target[selected_idx] += delta
            q_left_target[selected_idx] = np.clip(
                q_left_target[selected_idx],
                left_limits[selected_idx][0],
                left_limits[selected_idx][1]
            )
        else:
            q_right_target[selected_idx] += delta
            q_right_target[selected_idx] = np.clip(
                q_right_target[selected_idx],
                right_limits[selected_idx][0],
                right_limits[selected_idx][1]
            )

    def on_key(keycode):
        nonlocal selected_arm, selected_idx, q_left_target, q_right_target

        # --- 选择左臂关节 1-5 ---
        if keycode == ord('1'):
            selected_arm = "left"; selected_idx = 0
            print(f"选中: 左臂 关节1 ({LEFT_JOINT_NAMES[0]})  当前={q_left_target[0]:.3f} rad")
        elif keycode == ord('2'):
            selected_arm = "left"; selected_idx = 1
            print(f"选中: 左臂 关节2 ({LEFT_JOINT_NAMES[1]})  当前={q_left_target[1]:.3f} rad")
        elif keycode == ord('3'):
            selected_arm = "left"; selected_idx = 2
            print(f"选中: 左臂 关节3 ({LEFT_JOINT_NAMES[2]})  当前={q_left_target[2]:.3f} rad")
        elif keycode == ord('4'):
            selected_arm = "left"; selected_idx = 3
            print(f"选中: 左臂 关节4 ({LEFT_JOINT_NAMES[3]})  当前={q_left_target[3]:.3f} rad")
        elif keycode == ord('5'):
            selected_arm = "left"; selected_idx = 4
            print(f"选中: 左臂 关节5 ({LEFT_JOINT_NAMES[4]})  当前={q_left_target[4]:.3f} rad")

        # --- 选择右臂关节 6-0 ---
        elif keycode == ord('6'):
            selected_arm = "right"; selected_idx = 0
            print(f"选中: 右臂 关节1 ({RIGHT_JOINT_NAMES[0]})  当前={q_right_target[0]:.3f} rad")
        elif keycode == ord('7'):
            selected_arm = "right"; selected_idx = 1
            print(f"选中: 右臂 关节2 ({RIGHT_JOINT_NAMES[1]})  当前={q_right_target[1]:.3f} rad")
        elif keycode == ord('8'):
            selected_arm = "right"; selected_idx = 2
            print(f"选中: 右臂 关节3 ({RIGHT_JOINT_NAMES[2]})  当前={q_right_target[2]:.3f} rad")
        elif keycode == ord('9'):
            selected_arm = "right"; selected_idx = 3
            print(f"选中: 右臂 关节4 ({RIGHT_JOINT_NAMES[3]})  当前={q_right_target[3]:.3f} rad")
        elif keycode == ord('0'):
            selected_arm = "right"; selected_idx = 4
            print(f"选中: 右臂 关节5 ({RIGHT_JOINT_NAMES[4]})  当前={q_right_target[4]:.3f} rad")

        # --- 角度调整 ---
        elif keycode == ord('Q') or keycode == ord('q'):
            apply_delta(0.1)
            v = q_left_target[selected_idx] if selected_arm=='left' else q_right_target[selected_idx]
            print(f"  +0.1 -> {v:.3f}")
        elif keycode == ord('A') or keycode == ord('a'):
            apply_delta(-0.1)
            v = q_left_target[selected_idx] if selected_arm=='left' else q_right_target[selected_idx]
            print(f"  -0.1 -> {v:.3f}")
        elif keycode == ord('W') or keycode == ord('w'):
            apply_delta(0.05)
            v = q_left_target[selected_idx] if selected_arm=='left' else q_right_target[selected_idx]
            print(f"  +0.05 -> {v:.3f}")
        elif keycode == ord('S') or keycode == ord('s'):
            apply_delta(-0.05)
            v = q_left_target[selected_idx] if selected_arm=='left' else q_right_target[selected_idx]
            print(f"  -0.05 -> {v:.3f}")
        elif keycode == ord('E') or keycode == ord('e'):
            apply_delta(0.01)
            v = q_left_target[selected_idx] if selected_arm=='left' else q_right_target[selected_idx]
            print(f"  +0.01 -> {v:.3f}")
        elif keycode == ord('D') or keycode == ord('d'):
            apply_delta(-0.01)
            v = q_left_target[selected_idx] if selected_arm=='left' else q_right_target[selected_idx]
            print(f"  -0.01 -> {v:.3f}")

        # --- 打印 ---
        elif keycode == ord('P') or keycode == ord('p'):
            print("\n" + "="*60)
            print("  当前姿态（弧度）— 复制到 gate_action.py 对应位置")
            print("="*60)
            print(f"  POSE_XXX_L = np.array({np.round(q_left_target, 4).tolist()})")
            print(f"  POSE_XXX_R = np.array({np.round(q_right_target, 4).tolist()})")
            print(f"\n  角度(度): 左臂={np.round(np.degrees(q_left_target),1).tolist()}")
            print(f"            右臂={np.round(np.degrees(q_right_target),1).tolist()}")
            print("="*60 + "\n")

        # --- 重置 ---
        elif keycode == ord('R') or keycode == ord('r'):
            q_left_target[:] = [0.05, 0.15, -0.1, -0.2, 0.0]
            q_right_target[:] = [0.05, -0.15, 0.1, -0.2, 0.0]
            print("已重置")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        print("\n=== PM01 姿态调试工具 ===")
        print("1-5: 选左臂  |  6-0: 选右臂")
        print("Q/A: ±0.1  |  W/S: ±0.05  |  E/D: ±0.01")
        print("P: 打印角度  |  R: 重置  |  Esc: 退出")
        print("========================\n")

        while viewer.is_running():
            # ============== 双臂 PD 控制 ==============
            q_left = np.array([data.qpos[model.jnt_qposadr[j]] for j in left_joint_ids])
            qd_left = np.array([data.qvel[model.jnt_dofadr[j]] for j in left_joint_ids])
            q_right = np.array([data.qpos[model.jnt_qposadr[j]] for j in right_joint_ids])
            qd_right = np.array([data.qvel[model.jnt_dofadr[j]] for j in right_joint_ids])

            tau_left = KP * (q_left_target - q_left) - KD * qd_left
            tau_right = KP * (q_right_target - q_right) - KD * qd_right

            for i, aid in enumerate(left_act_ids):
                data.ctrl[aid] = np.clip(tau_left[i], model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1])
            for i, aid in enumerate(right_act_ids):
                data.ctrl[aid] = np.clip(tau_right[i], model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1])

            # ============== 固定基座 ==============
            data.qpos[0:7] = [0, 0, 0.82, 1, 0, 0, 0]
            data.qvel[0:6] = 0

            # ============== 硬固定下半身（腿、腰、头）==============
            for jid, qadr, vadr, q0 in body_joints:
                data.qpos[qadr] = q0
                data.qvel[vadr] = 0.0
                # 对应执行器也设为 0，避免力矩累积
                aid = find_actuator(jid)
                if aid >= 0:
                    data.ctrl[aid] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print("\n退出")


if __name__ == "__main__":
    main()
