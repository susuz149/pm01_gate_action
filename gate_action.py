"""
PM01 人形机器人 - 模拟开关闸门动作
任务书：人形机器人动作开发工程师技术考核任务书——PM01T800模拟开关闸门

阶段流程：1（初始）→ 2（抬起）→ 3（展开开闸）→ 4（收回关闸）→ 7（复位）
说明：进阶阶段 5（腕部旋转）、6（拨动开关）因模型无腕关节，未实现。

操作：
  Space - 暂停/继续
  R     - 重置
  S     - 停止（保持姿态）
  V     - 切换阶段提示
  Esc   - 退出
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path



# 配置


MODEL_PATH = Path(__file__).parent / "assets/resource/pm_v2.xml"

# 左臂关节（顺序：肩pitch, 肩roll, 肩yaw, 肘pitch, 肘yaw）
LEFT_JOINTS = ["J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
               "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L"]
# 右臂关节
RIGHT_JOINTS = ["J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R",
                "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R"]

# PD 控制参数（肩3 + 肘2）
KP = np.array([100.0, 100.0, 80.0, 80.0, 50.0])
KD = np.array([4.0, 4.0, 3.0, 3.0, 2.0])

# 各阶段持续时间（秒）
# 阶段1: 初始姿态停留时间
# 阶段2/3/4/7: 从起点姿态运动到终点姿态的时间
STAGE_DURATION = {
    1: 1.0,   # 初始姿态停留
    2: 1.5,   # 抬起至中心（运动时间）
    3: 1.5,   # 两侧展开-开闸（运动时间）
    4: 1.5,   # 返回中心-关闸（运动时间）
    7: 1.5,   # 平滑复位（运动时间）
}

# 关节速度上限（弧度/秒）
MAX_VEL = 3.0



# 各阶段目标姿态（弧度）—— 已通过 pose_tuner 调试

# 顺序: [肩pitch, 肩roll, 肩yaw, 肘pitch, 肘yaw]

# 阶段1 & 7终点：初始下垂
POSE_INIT_L  = np.array([0.05,  0.15, -0.1, -0.2, 0.0])
POSE_INIT_R  = np.array([0.05, -0.15,  0.1, -0.2, 0.0])

# 阶段2终点 & 阶段4终点：脸前中心
POSE_CENTER_L = np.array([-1.95,  1.05, -1.15, -1.8,  1.9])
POSE_CENTER_R = np.array([-1.95, -1.05,  1.15, -1.8, -1.9])

# 阶段3终点：两侧展开
POSE_OPEN_L = np.array([-2.05,  1.25, -1.05, -0.3626,  0.9])
POSE_OPEN_R = np.array([-2.05, -1.25,  1.05, -0.3626, -0.9])



# 五次多项式轨迹插值

def quintic(q0, q1, t, T):
    """
    起止速度/加速度均为 0 的五次多项式插值。
    q0: 起始角度, q1: 终止角度, t: 当前时间, T: 总时长
    返回: (当前角度, 当前角速度)
    """
    if T <= 0:
        return q1, np.zeros_like(q1)
    t = np.clip(t, 0.0, T)
    tau = t / T
    #q(tau)=a0+a1*tau+a2*tau^2+a3*tau^3+a4*tau^4+a5*tau^5
    #q`(tau)=a1+2*a2*tau+3*a3*tau^2+4*a4*tau^3+5*a5*tau^4
    # q``(tau)=2*a2+6*a3*tau+12*a4*tau^2+20*a5*tau^3
    # 由边界条件 q(0)=q0, q(T)=q1, q`(0)=q`(T)=0, q``(0)=q``(T)=0 可解得系数
    q = q0 + (q1 - q0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    #速度
    qd = (q1 - q0) * (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    return q, np.clip(qd, -MAX_VEL, MAX_VEL)



# 状态机

class StateMachine:
    """阶段状态机：1 → 2 → 3 → 4 → 7 → 1（循环）"""

    def __init__(self):
        self.stage = 1
        self.stage_t0 = 0.0
        self.cycle = 0
        self.paused = False
        self.stopped = False

    def reset(self, t):
        """重置到阶段 1"""
        self.stage = 1
        self.stage_t0 = t
        self.stopped = False
        self.cycle = 0

    def stop(self):
        """停止阶段切换，保持当前姿态"""
        self.stopped = True

    def toggle_pause(self):
        """切换暂停/继续，返回当前状态"""
        self.paused = not self.paused
        return self.paused

    def update(self, t):
        """每帧调用，检查是否需要切换阶段"""
        if self.paused or self.stopped:
            return
        if t - self.stage_t0 >= STAGE_DURATION[self.stage]:
            self._advance(t)

    def _advance(self, t):
        """推进到下一个阶段"""
        self.stage_t0 = t
        flow = {1: 2, 2: 3, 3: 4, 4: 7, 7: 1}
        self.stage = flow[self.stage]
        if self.stage == 1:
            self.cycle += 1

    def get_target(self, t):
        """返回 (左臂目标角度, 右臂目标角度, 左臂目标速度, 右臂目标速度)"""
        elapsed = t - self.stage_t0
        dur = STAGE_DURATION[self.stage]

        # (起点左, 终点左, 起点右, 终点右)
        pairs = {
            1: (POSE_INIT_L,   POSE_INIT_L,   POSE_INIT_R,   POSE_INIT_R),   # 保持
            2: (POSE_INIT_L,   POSE_CENTER_L, POSE_INIT_R,   POSE_CENTER_R), # 初始 → 中心
            3: (POSE_CENTER_L, POSE_OPEN_L,   POSE_CENTER_R, POSE_OPEN_R),   # 中心 → 展开
            4: (POSE_OPEN_L,   POSE_CENTER_L, POSE_OPEN_R,   POSE_CENTER_R), # 展开 → 中心
            7: (POSE_CENTER_L, POSE_INIT_L,   POSE_CENTER_R, POSE_INIT_R),   # 中心 → 初始
        }
        q0l, q1l, q0r, q1r = pairs[self.stage]
        ql, qdl = quintic(q0l, q1l, elapsed, dur)
        qr, qdr = quintic(q0r, q1r, elapsed, dur)
        return ql, qr, qdl, qdr

    def name(self):
        """当前阶段名称"""
        return {
            1: "阶段1 初始状态",
            2: "阶段2 双手抬起至中心",
            3: "阶段3 两侧展开(开闸)",
            4: "阶段4 返回中心(关闸)",
            7: "阶段7 平滑复位",
        }[self.stage]



# 主程序

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    #  查找关节和执行器ID 
    # joint关节名字翻译成ID
    left_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_JOINTS]
    right_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_JOINTS]
    #actuator_trnid用执行器当前的id和类型,类型不管,然后model.actuator_trnid[aid, 0]返回的值就是这个执行器对应的关节
    def find_actuator(jid):
        for aid in range(model.nu):
            if model.actuator_trnid[aid, 0] == jid:
                return aid
        return -1
    #第 jid 个关节的角度范围
    left_aids = [find_actuator(j) for j in left_jids]
    right_aids = [find_actuator(j) for j in right_jids]
    #关节限位返回一个二元数组`[最小值, 最大值]` （弧度）
    left_lims = [model.jnt_range[j] for j in left_jids]
    right_lims = [model.jnt_range[j] for j in right_jids]

    #  打印关节信息 
    print("=== 左臂关节 ===")
    for i, n in enumerate(LEFT_JOINTS):
        #`left_lims[0][0]` `-2.79` 第 0 个关节的最小值（第 0 个元素）
        print(f"  {n}: 限位=[{left_lims[i][0]:.3f}, {left_lims[i][1]:.3f}] rad")
    print("=== 右臂关节 ===")
    for i, n in enumerate(RIGHT_JOINTS):
        print(f"  {n}: 限位=[{right_lims[i][0]:.3f}, {right_lims[i][1]:.3f}] rad")

    #  初始化姿态并收集需固定的关节 
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.82   # 基座高度
    data.qpos[3] = 1.0    # 四元数 w 分量（初始朝向）

    # 双臂初始角度
    for i, jid in enumerate(left_jids):
        #jid在 qpos数组里的地址
        data.qpos[model.jnt_qposadr[jid]] = POSE_INIT_L[i]
        #才用这个地址去索引`data.qpos`
    for i, jid in enumerate(right_jids):
        data.qpos[model.jnt_qposadr[jid]] = POSE_INIT_R[i]

    # 收集非双臂关节（硬固定，保证下半身不动）
    arm_jid_set = set(left_jids + right_jids)
    body_joints = []
    for jid in range(model.njnt):
        if jid not in arm_jid_set:
            #位置数组索引地址
            qadr = model.jnt_qposadr[jid]
            #速度数组索引地址
            vadr = model.jnt_dofadr[jid]
            #记录这个关节的初始角度,固定
            q0 = data.qpos[qadr]
            body_joints.append((jid, qadr, vadr, q0))

    print(f"\n总关节: {model.njnt}, 双臂关节: {len(arm_jid_set)}, 固定关节: {len(body_joints)}")
    #告诉mujoco当前的状态数据已经更新了,需要重新计算
    mujoco.mj_forward(model, data)

    #  状态机与时间 
    fsm = StateMachine()
    sim_time = 0.0
    wall_start = time.time()
    last_print_stage = -1
    print_stage = True

    #  键盘回调 
    def on_key(keycode):
        nonlocal print_stage
        if keycode == 32:  # Space
            paused = fsm.toggle_pause()
            print(f"[按键] {'暂停' if paused else '继续'}", flush=True)
        elif keycode == ord('R') or keycode == ord('r'):
            fsm.reset(sim_time)
            print("[按键] 重置", flush=True)
        elif keycode == ord('S') or keycode == ord('s'):
            fsm.stop()
            print("[按键] 停止（保持当前姿态）", flush=True)
        elif keycode == ord('V') or keycode == ord('v'):
            print_stage = not print_stage
            print(f"[按键] 阶段提示: {'开' if print_stage else '关'}", flush=True)

    #  启动 viewer 并进入仿真循环 
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        print("\n" + "="*60, flush=True)
        print("  PM01 人形机器人 - 模拟开关闸门动作", flush=True)
        print("="*60, flush=True)
        print("  阶段流程: 1 → 2 → 3 → 4 → 7（基础必做全部完成）", flush=True)
        print("  控制方式: 关节空间 PD 控制 + 五次多项式轨迹插值", flush=True)
        print(f"  仿真步长: {model.opt.timestep*1000:.1f} ms", flush=True)
        print("-"*60, flush=True)
        print("  操作: Space=暂停/继续  R=重置  S=停止  V=切换阶段提示  Esc=退出", flush=True)
        print("="*60 + "\n", flush=True)

        while viewer.is_running():
            step_start = time.time()

            # —— 暂停或停止时，跳过所有计算和仿真，只刷新画面 ——
            if not fsm.paused and not fsm.stopped:

                # 1. 更新状态机
                fsm.update(sim_time)

                # 2. 阶段变化时打印
                if print_stage and fsm.stage != last_print_stage:
                    wall_time = time.time() - wall_start
                    print(f"[{wall_time:5.1f}s / {sim_time:5.1f}s] {fsm.name()}  (周期 {fsm.cycle+1})", flush=True)
                    last_print_stage = fsm.stage

                # 3. 计算当前步长目标姿态与速度
                qtl, qtr, qdtl, qdtr = fsm.get_target(sim_time)

                # 4. 限位保护
                qtl = np.clip(qtl, [l[0] for l in left_lims], [l[1] for l in left_lims])
                qtr = np.clip(qtr, [l[0] for l in right_lims], [l[1] for l in right_lims])

                # 5. 读取当前关节状态
                ql = np.array([data.qpos[model.jnt_qposadr[j]] for j in left_jids])
                qdl = np.array([data.qvel[model.jnt_dofadr[j]] for j in left_jids])
                qr = np.array([data.qpos[model.jnt_qposadr[j]] for j in right_jids])
                qdr = np.array([data.qvel[model.jnt_dofadr[j]] for j in right_jids])

                # 6. PD 控制计算力矩
                tau_l = KP * (qtl - ql) + KD * (qdtl - qdl)
                tau_r = KP * (qtr - qr) + KD * (qdtr - qdr)

                # 7. 限幅并输出到执行器
                for i, aid in enumerate(left_aids):
                    #data.ctrl是执行器控制信号数组
                    data.ctrl[aid] = np.clip(tau_l[i], model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1])
                for i, aid in enumerate(right_aids):
                    data.ctrl[aid] = np.clip(tau_r[i], model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1])

                # 8. 固定基座
                data.qpos[0:7] = [0, 0, 0.82, 1, 0, 0, 0]
                data.qvel[0:6] = 0

                # 9. 硬固定下半身（腿、腰、头）
                for jid, qadr, vadr, q0 in body_joints:
                    data.qpos[qadr] = q0
                    data.qvel[vadr] = 0.0
                    aid = find_actuator(jid)
                    if aid >= 0:
                        data.ctrl[aid] = 0.0

                # 10. 推进一步仿真
                # - 读取`data.ctrl` 里的控制力矩
                # - 计算所有接触力、重力、科里奥利力等
                # - 用动力学方程算每个关节的加速度
                # - 积分（加速度→速度→位置），得到新的 qvel 和 qpos
                # - 更新身体位置、接触、能量等所有数据
                mujoco.mj_step(model, data)
                sim_time += model.opt.timestep

            viewer.sync()

            # 限制到实时速度
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    wall_total = time.time() - wall_start
    print(f"\n程序结束。运行 {wall_total:.1f} 秒（真实）/ {sim_time:.1f} 秒（仿真），完成 {fsm.cycle} 个周期", flush=True)


if __name__ == "__main__":
    main()
