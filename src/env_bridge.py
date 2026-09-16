"""
env_bridge.py - cau noi UDP voi env.exe (mo phong bo xuong)

Giao thuc (UDP):
- env.exe bind port 5005 (nhan), gui state den 127.0.0.1:5006
- Python xin state   : gui [ -100.0 ]
- Python yeu cau ve  : gui [ -150.0 ]
- Reset skeleton     : gui [ -69.0, idx ]
- Impulse ngau nhien : gui [ -68.0, mag ]  (mag = 0 de tat)
- Gui action         : gui 10 float = targetAngle cua 10 khop
  Thu tu khop: head, hip, armR, armL, forearmR, forearmL, legR, legL, calfR, calfL

- env.exe tra ve state: 39 float:
  11 xuong * [sin(angle), cos(angle), angVel]
  Thu tu xuong: head, body, armL, armR, forearmL, forearmR, legL, legR, calfL, calfR, hip
  + hip.pos.y/600, hip.pos.x/800, hip.vel.y/600, hip.vel.x/800
  + calfL cham dat (1.0/0.0), calfR cham dat (1.0/0.0)
"""
import socket, struct, time, math

STATE_DIM = 39
ACTION_DIM = 10

# env.exe gui state den port 5006 -> phai bind dung port nay de nhan
state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_sock.bind(("127.0.0.1", 5006))
state_sock.settimeout(2.0)

# gui lenh/action den env.exe (bind port 5005) bang socket rieng
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

ENV_CMD = ("127.0.0.1", 5005)

# Tu the dung thang (target goc ban dau cua bo xuong, khong jitter)
# Thu tu khop: head, hip, armR, armL, forearmR, forearmL, legR, legL, calfR, calfL
STAND = [0.0, 4.95, 3.0, 4.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
JOINT_NAMES = ["head", "hip", "armR", "armL", "forearmR", "forearmL",
               "legR", "legL", "calfR", "calfL"]

# ---------------- Tham so gait (di bo bang dao dong sin) ----------------
GAIT_FREQ  = 1.5    # Hz - tan so buoc chan giay
SWING_LEG  = 0.35   # rad - bien do vung dui
SWING_KNEE = 0.45   # rad - bien do gap goi
SWING_ARM  = 0.25   # rad - bien do vung tay

# Bien do dao dong cho tung khop (0 = giu nguyen tu the STAND)
AMP = [0.0,           # head
       0.0,           # hip
       SWING_ARM,     # armR  - vung nguo voi chan ben phai
       SWING_ARM,     # armL  - vung nguo voi chan ben trai
       0.0,           # forearmR
       0.0,           # forearmL
       SWING_LEG,     # legR
       SWING_LEG,     # legL  - vung nguo pha voi legR
       SWING_KNEE,    # calfR - gap goi pha lech
       SWING_KNEE]    # calfL
PHASE = [0.0,           # head
         0.0,           # hip
         math.pi,       # armR
         0.0,           # armL
         0.0, 0.0,      # forearmR, forearmL
         0.0,           # legR
         math.pi,       # legL
         math.pi / 2,   # calfR
         3 * math.pi / 2]  # calfL

def send(*vals):
    cmd_sock.sendto(struct.pack(f"<{len(vals)}f", *vals), ENV_CMD)

def request_state():
    # luu y: env.exe bo qua goi tin < 2 float -> luon gui toi thieu 2 float
    send(-100.0, 0.0)
    data, _ = state_sock.recvfrom(STATE_DIM * 4)
    return struct.unpack(f"<{STATE_DIM}f", data)

first = True
frame_count = 0
import sys

def arg(name, default=None, cast=float):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default

max_frames = arg("--frames", None, int)
GAIT_FREQ  = arg("--freq",  GAIT_FREQ)
AMP_SCALE  = arg("--amp",   1.0)
MODE       = arg("--mode",  "walk", str)   # walk | stand
QUIET      = "--quiet" in sys.argv
try:
    while True:
        state = request_state()
        if first:
            hip_y, hip_x = state[33], state[34]
            print(f"Da nhan state tu env.exe: hip=({hip_x*800:.0f}, {hip_y*600:.0f})")
            first = False

        # ---- Sinh tham so dieu khien: gait di bo bang dao dong sin ----
        # action[i] = STAND[i] + AMP[i]*scale*sin(2*pi*f*t + PHASE[i])
        t = frame_count / 60.0  # thoi gian mo phong (giay)
        w = 2.0 * math.pi * GAIT_FREQ
        if MODE == "walk":
            action = [STAND[i] + AMP[i] * AMP_SCALE * math.sin(w * t + PHASE[i])
                      for i in range(ACTION_DIM)]
        else:
            action = list(STAND)  # mode stand: giu nguyen tu the dung

        # Gui action (10 targetAngle) ve env.exe de dinh hinh bo xuong
        cmd_sock.sendto(struct.pack(f"<{ACTION_DIM}f", *action), ENV_CMD)

        # In tham so da gui ve env.cpp (moi 15 frame, tat bang --quiet)
        if not QUIET and frame_count % 15 == 0:
            hip_y, hip_x = state[33], state[34]
            vals = " ".join(f"{n}={a:+.2f}" for n, a in zip(JOINT_NAMES, action))
            print(f"frame {frame_count:04d} | hip=({hip_x*800:.0f},{hip_y*600:.0f}) | {vals}")

        send(-150.0, 0.0)  # yeu cau env.exe ve khung hinh
        time.sleep(1 / 60)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            print(f"Hoan thanh {frame_count} frame state<->action")
            break
except KeyboardInterrupt:
    pass
except socket.timeout:
    print("Khong nhan duoc state tu env.exe - hay mo env.exe truoc!")

