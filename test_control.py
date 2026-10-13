"""
极简测试：确认 PD 控制和执行器是否正常工作
运行后双臂应该会缓慢抬起，如果不动说明控制有问题
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "assets/resource/pm_v2.xml"

LEFT_JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
]
RIGHT_JOINT_NAMES = [
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
]


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 打印模型基本信息
    print(f"关节总数: {model.njnt}")
    print(f"执行器总数: {model.nu}")
    print(f"执行器类型(0=motor): {[model.actuator_trntype[i] for i in range(model.nu)]}")

    # 找关节 ID
    left_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_JOINT_NAMES]
    right_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_JOINT_NAMES]
    print(f"\n左臂关节ID: {left_jids}")
    print(f"右臂关节ID: {right_jids}")

    # 找执行器（按关节ID匹配）
    def find_actuator(jid):
        for aid in range(model.nu):
            if model.actuator_trnid[aid, 0] == jid:
                return aid
        return -1

    left_aids = [find_actuator(j) for j in left_jids]
    right_aids = [find_actuator(j) for j in right_jids]
    print(f"左臂执行器ID: {left_aids}")
    print(f"右臂执行器ID: {right_aids}")

    # 打印执行器控制范围
    print("\n左臂执行器 ctrlrange:")
    for i, aid in enumerate(left_aids):
        print(f"  {LEFT_JOINT_NAMES[i]}: {model.actuator_ctrlrange[aid]}")
    print("右臂执行器 ctrlrange:")
    for i, aid in enumerate(right_aids):
        print(f"  {RIGHT_JOINT_NAMES[i]}: {model.actuator_ctrlrange[aid]}")

    # 关节限位
    print("\n左臂关节限位:")
    for i, jid in enumerate(left_jids):
        print(f"  {LEFT_JOINT_NAMES[i]}: {model.jnt_range[jid]}")

    # 初始化
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.82
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    # 打印初始关节角度
    print("\n初始左臂角度:")
    for i, jid in enumerate(left_jids):
        print(f"  {LEFT_JOINT_NAMES[i]}: {data.qpos[model.jnt_qposadr[jid]]:.4f} rad")

    # PD 参数（先给大一点，确保能动）
    KP = 200.0
    KD = 10.0

    # 目标：肩 pitch 弯曲 -1.0 rad（抬起来）
    target_shoulder_pitch = -1.0

    sim_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n=== 测试开始 ===")
        print(f"目标：肩 pitch 弯曲到 {target_shoulder_pitch} rad")
        print("如果手臂抬起来了，说明 PD 控制正常")
        print("按 Esc 退出\n")

        last_print = 0.0

        while viewer.is_running():
            # 目标角度随时间变化：0-2秒抬起，2-4秒放下，循环
            cycle = sim_time % 4.0
            if cycle < 2.0:
                # 抬起
                frac = cycle / 2.0
                target = -0.1 + frac * (target_shoulder_pitch - (-0.1))
            else:
                # 放下
                frac = (cycle - 2.0) / 2.0
                target = target_shoulder_pitch + frac * (-0.1 - target_shoulder_pitch)

            # 对左右臂的肩 pitch 都加控制
            for side_aids, side_jids in [(left_aids, left_jids), (right_aids, right_jids)]:
                for i in range(len(side_aids)):
                    jid = side_jids[i]
                    aid = side_aids[i]
                    q = data.qpos[model.jnt_qposadr[jid]]
                    qd = data.qvel[model.jnt_dofadr[jid]]

                    # 第0个关节是肩pitch，让它动；其他保持0
                    if i == 0:
                        q_target = target
                    else:
                        q_target = 0.0

                    tau = KP * (q_target - q) - KD * qd
                    tau = np.clip(tau, model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1])
                    data.ctrl[aid] = tau

            # 固定基座
            data.qpos[0:7] = [0, 0, 0.82, 1, 0, 0, 0]
            data.qvel[0:6] = 0

            # 腿部也简单固定
            for jid in range(12):
                aid = find_actuator(jid)
                if aid >= 0:
                    qadr = model.jnt_qposadr[jid]
                    vadr = model.jnt_dofadr[jid]
                    target_q = 0.24 if jid == 1 else (-0.24 if jid == 7 else 0.0)
                    tau = 100.0 * (target_q - data.qpos[qadr]) - 5.0 * data.qvel[vadr]
                    data.ctrl[aid] = np.clip(tau, model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1])

            # 腰部头部固定
            for jid in [12, 23]:
                aid = find_actuator(jid)
                if aid >= 0:
                    qadr = model.jnt_qposadr[jid]
                    vadr = model.jnt_dofadr[jid]
                    tau = 50.0 * (0.0 - data.qpos[qadr]) - 2.0 * data.qvel[vadr]
                    data.ctrl[aid] = np.clip(tau, model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1])

            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            # 每秒打印一次
            if sim_time - last_print >= 1.0:
                q0 = data.qpos[model.jnt_qposadr[left_jids[0]]]
                print(f"t={sim_time:.1f}s  左肩pitch={q0:.3f} rad  目标={target:.3f}  误差={target-q0:.3f}")
                last_print = sim_time

            viewer.sync()
            time.sleep(model.opt.timestep)

    print("\n测试结束")


if __name__ == "__main__":
    main()
