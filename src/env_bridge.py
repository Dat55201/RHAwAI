"""
env_bridge.py - Cau noi UDP (Python <-> env.exe) + HUAN LUYEN REINFORCEMENT LEARNING
====================================================================================

CHUONG TRINH NAY LAM GI:
  1. Ket noi toi mo phong vat ly C++ (env.exe) qua UDP.
  2. Dung mang MLP 2 lop an (PyTorch) lam Policy dieu khien 10 khop cua bo xuong.
  3. Tu hoc 3 ky nang: giu thang bang (balance), dung thang dung (upright),
     vuon cao nhat co the (maximize height).
  4. Toi uu trong so bang loss.backward() + optimizer Adam
     (Policy Gradient: A2C per-step hoac REINFORCE theo episode).

------------------------------------------------------------------------------------
GIAO THUC UDP (khop voi DataManager trong src/env.cpp - KHONG can sua env.cpp)
------------------------------------------------------------------------------------
  env.exe bind port 5005 (nhan lenh), gui state ve 127.0.0.1:5006.
  Python -> env.exe:
    [-100.0, 0.0]   xin state (env tra loi bang 39 float)
    [-150.0, 0.0]   yeu cau ve 1 frame (glfwSwapBuffers)
    [ -69.0, idx ]  reset skeleton
    [ -68.0, mag ]  dat impulse_max (luc day ngau nhien; 0 = tat)
    [ 10 float   ]  targetAngle cho 10 khop (thu tu joints)
  env.exe -> Python (39 float):
    11 xuong * [sin(angle), cos(angle), angVel]
    + hip.pos.y/600, hip.pos.x/800, hip.vel.y/600, hip.vel.x/800
    + calfL cham dat (1.0/0.0), calfR cham dat (1.0/0.0)

------------------------------------------------------------------------------------
BAN DO 39 FLOAT STATE (thu tu bones trong env.cpp: head, body, armL, armR,
forearmL, forearmR, legL, legR, calfL, calfR, hip)
------------------------------------------------------------------------------------
    idx | bone      | sin | cos | angVel      idx | bone      | sin | cos | angVel
   -----+-----------+-----+-----+--------   -----+-----------+-----+-----+--------
     0  | head      |  0  |  1  |   2         18 | legL      | 18  | 19  |  20
     3  | body      |  3  |  4  |   5         21 | legR      | 21  | 22  |  23
     6  | armL      |  6  |  7  |   8         24 | calfL     | 24  | 25  |  26
     9  | armR      |  9  | 10  |  11         27 | calfR     | 27  | 28  |  29
    12  | forearmL  | 12  | 13  |  14         30 | hip       | 30  | 31  |  32
    15  | forearmR  | 15  | 16  |  17
   --------------------------------------------------------------------------------
    33 = hip.pos.y/600 (DO CAO)      35 = hip.vel.y/600
    34 = hip.pos.x/800               36 = hip.vel.x/800
    37 = calfL cham dat              38 = calfR cham dat

  * LUU Y: state KHONG chua toa do tuyet doi cua dau/than, nen "do cao" duoc lay
    bang hip.pos.y (xap xi trong tam co the - center of mass).
  * body (song lung) bat dau voi angle = 1.50 rad => sin(body_angle) = 1.0 chinh la
    tu the "song lung song song truc thang dung". Day la co so cua Upright Reward.

------------------------------------------------------------------------------------
THU TU 10 KHOP (action) - trung voi vector joints trong env.cpp va STAND ben duoi
------------------------------------------------------------------------------------
    head, hip, armR, armL, forearmR, forearmL, legR, legL, calfR, calfL
  targetAngle la GOC TUYET DOI (rad) giua 2 xuong, bien do thuc te 0 -> ~5 rad.

------------------------------------------------------------------------------------
CACH DUNG
------------------------------------------------------------------------------------
  # 0) Mo env.exe truoc (bat buoc - no la server UDP)
  # 1) HUAN LUYEN DUNG 1 GIO (het gio tu dong dung va luu checkpoint) - khuyen nghi
  python src/env_bridge.py --mode train --time 1h --no-render --save-every 25 --log-every 100
  # 1b) CHAY LIEN TUC 10 GIO KHONG NGHI (tu luu dinh ky + tu thu lai khi env.exe
  #     mat ket noi tam thoi trong han --retry, mac dinh 120s)
  python src/env_bridge.py --mode train --time 10h --no-render --save-every 20 --log-every 200
  # 2) Chay / trinh dien mo hinh da hoc lien tuc trong 1 gio
  python src/env_bridge.py --mode eval --resume rha_policy.pt --time 1h --no-render
  # 3) Huan luyen theo SO EPISODE thay vi theo thoi gian
  python src/env_bridge.py --mode train --episodes 200 --steps 300
  # 4) Chay thu policy da hoc (khong cap nhat trong so)
  python src/env_bridge.py --mode eval --resume rha_policy.pt --steps 600
  # 5) Che do cu (tuong thich nguoc): di bo bang dao dong sin / dung yen
  python src/env_bridge.py --mode walk --freq 1.5 --amp 1.0 --frames 600
  python src/env_bridge.py --mode stand

  LUU Y VE THOI GIAN:
  * --time nhan '1h', '30m', '90s', '1.5h', hoac so khong don vi (= PHUT).
  * --time 10h = chay LIEN TUC 10 GIO (36.000 giay ~ 2.16 trieu buoc vat ly).
  * --episodes van co the dung cung luc lam GIOI HAN CUNG (dung o cai nao toi truoc).
  * Phien dai: mat ket noi env.exe tam thoi se duoc THU LAI (han --retry) thay vi
    thoat; checkpoint luu dinh ky + ban backup '.bak' truoc moi lan ghi de.
  * Mo phong chay KHOA NHIP thoi gian thuc (~60 buoc vat ly/giay, dt=1/60) nen 1 gio
    thuc = ~216.000 buoc vat ly = ~1 gio mo phong. Dung --no-render + nhan phim Y
    trong cua so env.exe de tang len ~8 lan.

  MEO TANG TOC: them --no-render VA nhan phim Y trong cua so env.exe (frameInterval
  = 8) => vong lap C++ chay ~8x nhanh hon. Khi do moi buoc Python tuong ung ~8 buoc
  vat ly (dt hieu dung = 8/60 s), nghia la action duoc lap lai 8 lan.
"""
import argparse
import collections
import math
import os
import socket
import struct
import sys
import time

# ------------------------------------------------------------------------------------
# PyTorch - bat buoc cho --mode train / --mode eval
# ------------------------------------------------------------------------------------
TORCH_HINT = (
    "Chua cai PyTorch. Hay chay lenh sau roi thu lai:\n"
    "    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
)
try:
    # torch canh bao "Failed to initialize NumPy" khi chua cai numpy - vo hai vi
    # chuong trinh nay chi dung torch + thu vien chuan, khong dung numpy.
    import warnings

    warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:                        # van cho phep --mode walk/stand chay binh thuong
    torch = nn = F = Normal = None
    TORCH_AVAILABLE = False


def require_torch():
    """Bao loi ro rang neu nguoi dung chua cai PyTorch."""
    if not TORCH_AVAILABLE:
        raise SystemExit(TORCH_HINT)


# ------------------------------------------------------------------------------------
# Hang so giao thuc / observation
# ------------------------------------------------------------------------------------
STATE_DIM = 39                          # so float env.exe gui ve
ACTION_DIM = 10                         # so khop dieu khien duoc
ENV_CMD_ADDR = ("127.0.0.1", 5005)      # env.exe nhan lenh tai day
PY_STATE_ADDR = ("127.0.0.1", 5006)     # Python nhan state tai day
SOCKET_TIMEOUT = 2.0                    # giay - cho state toi da truoc khi bao loi
FRAME_DT = 1.0 / 60.0                   # env.cpp dung dt = 1/60 co dinh cho moi buoc vat ly

# Thu tu xuong trong state (khop bones[] cua env.cpp)
BONE_NAMES = ["head", "body", "armL", "armR", "forearmL", "forearmR",
              "legL", "legR", "calfL", "calfR", "hip"]
BONE_IDX = {name: i * 3 for i, name in enumerate(BONE_NAMES)}   # -> idx cua sin(angle)

IDX_HIP_POS_Y = 33       # = hip.pos.y / 600   (0..1)   -> DO CAO (height)
IDX_HIP_POS_X = 34       # = hip.pos.x / 800   (0..1)
IDX_HIP_VEL_Y = 35       # = hip.vel.y / 600
IDX_HIP_VEL_X = 36       # = hip.vel.x / 800
IDX_GROUND_L = 37        # calfL cham dat (1.0/0.0)
IDX_GROUND_R = 38        # calfR cham dat (1.0/0.0)

# Thu tu khop (action) + tu the dung thang goc (goc target ban dau cua env.cpp)
JOINT_NAMES = ["head", "hip", "armR", "armL", "forearmR", "forearmL",
               "legR", "legL", "calfR", "calfL"]
STAND = [0.0, 4.95, 3.0, 4.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]

# Chuan hoa observation cho mang neural: moi gia tri ep ve [-1, 1]
ANG_VEL_SCALE = 20.0     # env.cpp clamp angVel vao [-50, 50] -> chia 20 la dep

CMD_STATE = -100.0       # xin state
CMD_RENDER = -150.0      # ve 1 frame
CMD_RESET = -69.0        # reset skeleton
CMD_IMPULSE = -68.0      # dat impulse_max (0 = tat luc day ngau nhien)

# ------------------------------------------------------------------------------------
# Trong so ham thuong (Reward shaping) - chinh tai day neu muon doi "tinh cach" policy
# ------------------------------------------------------------------------------------
W_HEIGHT = 1.0           # thuong theo do cao hip.pos.y
W_UPRIGHT = 0.6          # thuong khi song lung thang dung
W_STABILITY = 0.15       # Phat rung lac (angVel) va van toc hip
W_BALANCE = 0.3          # thuong khi 2 chan cham dat (phat khi khong chan nao)
W_CENTER = 0.15          # phat khi hip troi xa vi tri xuat phat theo truc X
W_ALIVE = 0.10           # thuong song sot moi buoc
FALL_PENALTY = 10.0      # phat nang khi nga
FALL_HEIGHT = 0.12       # hip.pos.y/600 < nguong nay => coi la NGA, ket thuc episode
TILT_TOL_DEG = 45.0      # song lung nghieng qua nguong nay => 0 diem "thang dung"
                         # (dung goc nghieng thay vi sin: sin(20 do) = 0.94 -> gan nhu
                         #  khong phat, trong khi yeu cau la "giam thieu goc nghieng")

# Do cao chuan - DO THUC NGHIEM bang che do `--mode stand`:
#   o tu the STAND, bo xuong dung yen voi hip.pos.y/600 ~ 0.169 va 2 chan cham dat;
#   MOI bien the "thang chan/doi chan" deu chi cho 0.157..0.173 => chan gan nhu da
#   duoi het, bo xuong KHONG the cao hon ~0.175 khi con dung tren mat dat.
#
# Vi vay "vuon cao" duoc thuong theo ham bao hoa:
#       height_unit = clip((hip_y - HEIGHT_FLOOR) / (HEIGHT_REF - HEIGHT_FLOOR), 0, 1)
#   - 0.0 khi hip_y <= HEIGHT_FLOOR  (dang sup/nga)
#   - 1.0 khi hip_y >= HEIGHT_REF    (da vuon het tam voi cua bo xuong)
#   Neu de tran thuong CAO HON tam voi that (vi du +0.02 nhu ban dau), policy se bi
#   "lua": nang mot chan/nhay len de an them diem do cao (mat -0.3 diem thang bang
#   nhung duoc +0.8 diem do cao) => dung 1 chan roi nga. Do la ly do dung bao hoa.
HEIGHT_FLOOR = 0.13      # hip_y duoi nguong nay coi nhu dang sup xuong (0 diem do cao)
HEIGHT_REF = 0.17        # hip_y >= nguong nay = da vuon het tam voi (diem toi da)

# ====================================================================================
# 1) TANG GIAO TIEP UDP (Python <-> env.exe)
# ====================================================================================
class EnvNotRunning(RuntimeError):
    """Khong nhan duoc state tu env.exe (mo env.exe truoc khi chay Python)."""


state_sock = None      # socket nhan state (bind 127.0.0.1:5006)
cmd_sock = None        # socket gui lenh/action den env.exe (127.0.0.1:5005)


def open_sockets():
    """Tao + bind 2 socket UDP. Goi 1 lan luc khoi dong."""
    global state_sock, cmd_sock
    if state_sock is None:
        state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_sock.bind(PY_STATE_ADDR)      # env.exe gui state den port nay
        state_sock.settimeout(SOCKET_TIMEOUT)
    if cmd_sock is None:
        cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return state_sock, cmd_sock


def close_sockets():
    """Dong socket khi ket thuc."""
    global state_sock, cmd_sock
    for s in (state_sock, cmd_sock):
        if s is not None:
            try:
                s.close()
            except OSError:
                pass
    state_sock = cmd_sock = None


def send(*vals):
    """Gui mot goi float den env.exe (toi thieu 2 float - env.cpp bo qua goi ngan hon)."""
    cmd_sock.sendto(struct.pack("<%df" % len(vals), *vals), ENV_CMD_ADDR)


def _drain_state_socket():
    """
    Xoa het state cu con ket trong buffer truoc khi xin state moi.
    Neu Python chay cham hon env.exe, buffer co the chua nhieu state cu ->
    phai drain de reward luon tuong ung voi action vua gui.
    """
    state_sock.setblocking(False)
    try:
        while True:
            state_sock.recvfrom(STATE_DIM * 4)
    except OSError:
        pass                                # het du lieu -> BlockingIOError
    finally:
        state_sock.setblocking(True)
        state_sock.settimeout(SOCKET_TIMEOUT)


def request_state():
    """Xin state moi nhat tu env.exe -> tuple 39 float."""
    _drain_state_socket()
    send(CMD_STATE, 0.0)
    try:
        data, _ = state_sock.recvfrom(STATE_DIM * 4)
    except (socket.timeout, OSError) as exc:
        raise EnvNotRunning(
            "Khong nhan duoc state tu env.exe - hay mo env.exe truoc! "
            "(kiem tra cua so env.exe da bat va phim P khong o trang thai UDP OFF)"
        ) from exc
    if len(data) < STATE_DIM * 4:
        return request_state()              # goi tin le -> xin lai
    return struct.unpack("<%df" % STATE_DIM, data[: STATE_DIM * 4])


def request_state_retry(retry_seconds=15.0):
    """
    request_state() + TU DONG THU LAI - bat buoc cho phien dai (vi du chay 10h).

    Neu env.exe mat ket noi TAM THOI (may giat, dia ban, update Windows...),
    ham nay khong cho phep chuong trinh chet ngay:
      - in [CANH BAO] lan dau mat ket noi,
      - thu lai moi 2 giay toi da `retry_seconds` giay,
      - khi env.exe tro lai -> in [PHUC HOI] va tra ve state (phien chay tiep
        lien mach, khong mat du lieu chi vi 1 giay mat ket noi).
    Qua han retry ma van mat ket noi -> raise EnvNotRunning (thoat co kiem soat,
    checkpoint van duoc luu trong finally).
    """
    retry_seconds = max(0.0, float(retry_seconds))
    outage_start = None
    while True:
        try:
            state = request_state()
            if outage_start is not None:
                log("[PHUC HOI] tiep tuc huan luyen sau %.1f giay mat ket noi env.exe."
                    % (time.time() - outage_start))
            return state
        except EnvNotRunning as exc:
            if outage_start is None:
                outage_start = time.time()
                log("[CANH BAO] mat ket noi env.exe: %s" % exc)
            remaining = retry_seconds - (time.time() - outage_start)
            if remaining <= 0:
                raise                       # qua han -> thoat co kiem soat
            log("         thu lai sau 2 giay... (con %.0fs trong han --retry)" % remaining)
            time.sleep(2.0)


def reset_env():
    """Reset bo xuong ve tu the goc (env.cpp goi sk->reset())."""
    send(CMD_RESET, 0.0)


def set_impulse(mag):
    """Bat/tat luc day ngau nhien (impulse_max trong env.cpp). mag = 0 de tat."""
    send(CMD_IMPULSE, float(mag))


def render_tick(enabled=True):
    """Yeu cau env.exe ve 1 frame (glfwSwapBuffers) neu cho phep."""
    if enabled:
        send(CMD_RENDER, 0.0)


# ====================================================================================
# 2) CHUAN HOA STATE  (moi gia tri -> [-1, 1], cung triet ly voi ham tanh cua mang)
# ====================================================================================
def normalize_state(raw, device=None):
    """
    Chuyen 39 float tho tu env.exe thanh tensor float32 da chuan hoa.

    Quy tac:
      - sin/cos (index %3 != 2) da nam trong [-1, 1] -> giu nguyen.
      - angVel (index %3 == 2) chia ANG_VEL_SCALE roi clamp [-1, 1].
      - hip pos/vel: y/600 va x/800 da chuan hoa san -> clamp [-1, 1].
      - grounded: dua ve dung 0.0 hoac 1.0.
    """
    s = list(raw[:STATE_DIM])
    for i in range(len(BONE_NAMES)):                     # 11 xuong * 3
        j = i * 3 + 2                                    # vi tri angVel
        s[j] = max(-1.0, min(1.0, s[j] / ANG_VEL_SCALE))
    s[IDX_HIP_POS_Y] = max(0.0, min(1.0, s[IDX_HIP_POS_Y]))
    s[IDX_HIP_POS_X] = max(0.0, min(1.0, s[IDX_HIP_POS_X]))
    s[IDX_HIP_VEL_Y] = max(-1.0, min(1.0, s[IDX_HIP_VEL_Y]))
    s[IDX_HIP_VEL_X] = max(-1.0, min(1.0, s[IDX_HIP_VEL_X]))
    s[IDX_GROUND_L] = 1.0 if s[IDX_GROUND_L] > 0.5 else 0.0
    s[IDX_GROUND_R] = 1.0 if s[IDX_GROUND_R] > 0.5 else 0.0
    t = torch.tensor(s, dtype=torch.float32)
    return t.to(device) if device is not None else t


# ====================================================================================
# 3) HAM THUONG (REWARD FUNCTION) - dinh nghia "tu the tot"
# ====================================================================================
def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def compute_reward(raw, x_home=0.5):
    """
    Tinh reward tu 1 state tho (39 float) cua env.exe.

    Tra ve: (reward: float, done: bool, info: dict)

    Cong thuc:
        height    = clip((hip.pos.y/600 - HEIGHT_FLOOR) / (HEIGHT_REF - HEIGHT_FLOOR), 0, 1)
                    -> 0 khi hip sut thap (sap nga), 1.0 khi hip da vuon het tam voi
                       (do thuc nghiem bang --mode stand, xem chu thich HEIGHT_REF)
        upright   = max(0, 1 - tilt/45 do)  voi tilt = acos(sin(body_angle))
                    -> 1.0 khi song lung thang dung tuyet doi, 0 khi nghieng >= 45 do
        stability = -(|angVel| trung binh + |hip.vx| + |hip.vy|)  -> cang it rung lac cang tot
        balance   = +1.0 neu CA HAI chan cham dat,
                     0.0 neu 1 chan,
                    -0.5 neu khong chan nao cham dat
        center    = -|hip.pos.x/800 - x_home|               -> khong troi ngang
        alive     = +1.0 moi buoc song sot
        fall      = -FALL_PENALTY va done=True neu hip_y < FALL_HEIGHT

        reward = W_HEIGHT*height + W_UPRIGHT*upright + W_STABILITY*stability
               + W_BALANCE*balance + W_CENTER*center + W_ALIVE*alive (+ fall)

    x_home: vi tri X chuan hoa cua hip luc bat dau episode (moc "dung yen 1 cho").
    """
    hip_y = _clamp01(raw[IDX_HIP_POS_Y])                  # 0..1  (do cao)
    hip_x = raw[IDX_HIP_POS_X]
    hip_vy = raw[IDX_HIP_VEL_Y]
    hip_vx = raw[IDX_HIP_VEL_X]

    body_sin = raw[BONE_IDX["body"]]                      # sin(angle song lung) -> index 3

    # --- 1. Height Reward -------------------------------------------------------
    # Thuong bao hoa: 0 khi hip_y <= HEIGHT_FLOOR, len dan den 1.0 tai HEIGHT_REF
    # (xem chu thich HEIGHT_FLOOR / HEIGHT_REF o dau file de biet ly do bao hoa).
    height_unit = (hip_y - HEIGHT_FLOOR) / (HEIGHT_REF - HEIGHT_FLOOR)
    height_unit = max(0.0, min(1.0, height_unit))
    r_height = W_HEIGHT * height_unit

    # --- 2. Upright Reward -----------------------------------------------------
    # Song lung thang dung <=> goc nghieng so voi truc thang dung = 0.
    # tilt = acos(sin(body_angle)) = goc lech cua truc xuong so voi phuong thang dung.
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, body_sin))))
    upright = max(0.0, 1.0 - tilt_deg / TILT_TOL_DEG)
    r_upright = W_UPRIGHT * upright

    # --- 3. Stability / Balance Reward (it rung lac, it dich chuyen) -----------
    ang_vel_sum = sum(abs(raw[i * 3 + 2]) for i in range(len(BONE_NAMES)))
    shaking = min(1.0, (ang_vel_sum / len(BONE_NAMES)) / ANG_VEL_SCALE)   # trung binh |angVel|
    drift = min(1.0, abs(hip_vx))                        # truot ngang
    bounce = min(1.0, abs(hip_vy))                       # nhun len xuong
    stability = -(shaking + drift + bounce)
    r_stability = W_STABILITY * stability

    # --- 4. Balance Reward (chan cham dat) -------------------------------------
    ground_l = raw[IDX_GROUND_L] > 0.5
    ground_r = raw[IDX_GROUND_R] > 0.5
    if ground_l and ground_r:
        balance = 1.0
    elif ground_l or ground_r:
        balance = 0.0
    else:
        balance = -0.5                                   # ca 2 chan roi khoi mat dat
    r_balance = W_BALANCE * balance

    # --- 5. Center Reward (khong troi ngang khoi cho cu) -----------------------
    r_center = -W_CENTER * min(1.0, abs(hip_x - x_home))

    # --- 6. Alive Bonus --------------------------------------------------------
    r_alive = W_ALIVE

    total = r_height + r_upright + r_stability + r_balance + r_center + r_alive

    # --- 7. Fall Penalty + ket thuc episode ------------------------------------
    done = False
    fall = 0.0
    if hip_y < FALL_HEIGHT:
        total -= FALL_PENALTY
        fall = -FALL_PENALTY
        done = True                                      # bo xuong nga -> het episode

    info = {
        "height": r_height, "upright": r_upright, "stability": r_stability,
        "balance": r_balance, "center": r_center, "alive": r_alive, "fall": fall,
        "total": total, "hip_y": hip_y, "hip_x": hip_x, "tilt_deg": tilt_deg,
        "grounded": int(ground_l) + int(ground_r), "done": done,
        "height_unit": height_unit, "upright_unit": upright,
    }
    return total, done, info


def fmt_reward(info):
    """In gon cac thanh phan reward de log de doc."""
    return ("H=%.2f U=%.2f S=%+.2f B=%+.2f C=%+.2f A=%.2f%s"
            % (info["height"], info["upright"], info["stability"], info["balance"],
               info["center"], info["alive"],
               (" FALL=%+.1f" % info["fall"]) if info["fall"] else ""))


# ====================================================================================
# 4) TIEN ICH THONG KE / LOG
# ====================================================================================
class RunningStat:
    """Uoc luong mean/std truc tuyen (Welford) - dung de chuan hoa advantage."""

    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.eps = eps

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    @property
    def std(self):
        return math.sqrt(max(self.var, 0.0)) + self.eps


class Window:
    """Gia tri trung binh cua N phan tu gan nhat (de bao cao tien do hoc)."""

    def __init__(self, size):
        self.buf = collections.deque(maxlen=size)

    def add(self, v):
        self.buf.append(v)

    @property
    def mean(self):
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

    @property
    def last(self):
        return self.buf[-1] if self.buf else 0.0


def log(msg):
    """In log co flush ngay de theo doi tien do huan luyen theo thoi gian thuc."""
    print(msg, flush=True)


# ====================================================================================
# 4b) NGAN SACH THOI GIAN (wall-clock) - de chay "dung 1 gio" thay vi dem episode
# ====================================================================================
def parse_duration(text):
    """
    Doi chuoi thoi gian thanh SO GIAY.
        '1h' -> 3600 | '30m' -> 1800 | '90s' -> 90 | '1.5h' -> 5400
        '45'  (khong co don vi) -> 45 PHUT
    Nem SystemExit voi thong bao ro rang neu chuoi sai.
    """
    raw = str(text).strip().lower()
    units = {"s": 1.0, "sec": 1.0, "giay": 1.0,
             "m": 60.0, "min": 60.0, "phut": 60.0,
             "h": 3600.0, "hr": 3600.0, "gio": 3600.0}
    unit = None
    for suffix in sorted(units, key=len, reverse=True):      # khop hau to dai truoc
        if raw.endswith(suffix) and raw[: -len(suffix)].strip():
            unit, raw = units[suffix], raw[: -len(suffix)].strip()
            break
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit("[LOI] --time khong hop le: '%s' (vi du dung: 1h, 30m, 90s, 45)"
                         % text)
    if value <= 0:
        raise SystemExit("[LOI] --time phai > 0 (ban nhap '%s')" % text)
    return value * (unit if unit is not None else units["m"])    # khong don vi = PHUT


def hhmmss(seconds):
    """3600 -> '01:00:00' (dung cho log tien do)."""
    seconds = max(0, int(round(seconds)))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


class TimeBudget:
    """
    Ngan sach thoi gian THUC (wall-clock) cho mot phien chay.
    Tra loi "con bao lau nua thi het 1 gio?" va cho phep dung DUNG HAN.
    """

    def __init__(self, seconds):
        self.total = float(seconds)
        self.t0 = time.time()

    @property
    def elapsed(self):
        return time.time() - self.t0

    @property
    def remaining(self):
        return max(0.0, self.total - self.elapsed)

    def expired(self):
        """True khi da dung het ngan sach thoi gian."""
        return self.elapsed >= self.total

    def end_clock(self):
        """Gio tuong (HH:MM:SS) luc se ket thuc."""
        return time.strftime("%H:%M:%S", time.localtime(self.t0 + self.total))

    def progress_line(self, extra=""):
        """Dong [TIEN DO] de in dinh ky trong phien dai (vi du 1 gio)."""
        pct = 100.0 * min(1.0, self.elapsed / self.total) if self.total > 0 else 100.0
        return ("[TIEN DO] %s / %s (%.1f%%) | con lai %s | ket thuc ~ %s%s"
                % (hhmmss(self.elapsed), hhmmss(self.total), pct,
                   hhmmss(self.remaining), self.end_clock(), (" | " + extra) if extra else ""))


# ====================================================================================
# 5) MANG NEURAL (POLICY NETWORK) - 2 LOP AN, TANH XUYEN SUOT
# ====================================================================================
if TORCH_AVAILABLE:

    class PolicyNet(nn.Module):
        """
        MLP 2 lop an lam Policy (actor) + Critic (value baseline), dung chung trunk.

            input  : 39 float state da chuan hoa ve [-1, 1]
            hidden : Linear(39 -> 128) + tanh  ->  Linear(128 -> 128) + tanh
            heads  : mu = tanh(Linear(128 -> 10))      -> action trong (-1, 1)
                     value = Linear(128 -> 1)          -> V(s) (baseline)

        Phan phoi chinh sach la "squashed Gaussian":
            u ~ Normal(mu, sigma) ;  a = tanh(u)
            log pi(a|s) = log N(u) - log(1 - a^2 + eps)
        => vua giu a trong [-1, 1] (dung tinh than tanh), vua tinh duoc log-prob
           chinh xac de lan truyen nguoc bang loss.backward().
        """

        def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=128,
                     log_std_init=-1.4, log_std_min=-3.5, log_std_max=-1.0):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden)         # lop an 1
            self.fc2 = nn.Linear(hidden, hidden)            # lop an 2
            self.mu_head = nn.Linear(hidden, action_dim)    # policy head
            self.value_head = nn.Linear(hidden, 1)          # critic head
            # Do lech chuan la tham so HOC DUOC (khong phu thuoc state) -> don gian, on dinh.
            # LUU Y: day la nhieu kham pha tren TUNG khop (rad). Bai toan giu thang bang
            # can nhieu NHO (mu~0.25 rad) - neu de sigma ~ 0.5-1.0 rad thi bo xuong bi
            # lac lien tuc va khong bao gio hoc duoc tu the on dinh.
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max
            self._init_weights()

        def _init_weights(self):
            """Khoi tao kieu orthogonal - on dinh cho mang dung tanh."""
            for layer in (self.fc1, self.fc2):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
            # gain nho cho mu_head => ban dau mu ~ 0 => action ~ STAND (tu the dung biet truoc)
            nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)

        def forward(self, state):
            """-> (mu, log_std, value)."""
            h = torch.tanh(self.fc1(state))
            h = torch.tanh(self.fc2(h))
            mu = torch.tanh(self.mu_head(h))                # nen ve (-1, 1)
            value = self.value_head(h).squeeze(-1)          # V(s)
            log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
            return mu, log_std, value

        def act(self, state, deterministic=False):
            """
            Lay mau action. Tra ve (a, log_prob, value, entropy).
            deterministic=True -> a = tanh(mu) (dung khi eval / chay thu).
            """
            mu, log_std, value = self.forward(state)
            dist = Normal(mu, log_std.exp())
            if deterministic:
                a = torch.tanh(mu)
                zero = torch.zeros((), device=state.device)
                return a, zero, value, zero
            # detach() => dung score-function estimator (REINFORCE) cho log_prob,
            # tranh tron lan gradient cua duong reparameterize.
            u = dist.rsample().detach()
            a = torch.tanh(u)
            log_prob = (dist.log_prob(u) - torch.log(1.0 - a * a + 1e-6)).sum(-1)
            entropy = dist.entropy().sum(-1)
            return a, log_prob, value, entropy

else:                                   # stub: bao loi ro rang neu thieu PyTorch
    class PolicyNet:
        def __init__(self, *args, **kwargs):
            raise SystemExit(TORCH_HINT)


def joint_target_from_action(a, stand=None, action_scale=None):
    """
    Doi vector action a in (-1, 1) thanh 10 targetAngle gui ve env.exe.
        targetAngle[i] = STAND[i] + action_scale * a[i]
    => policy khoi dau gan nhu dung o tu the STAND va tu tim kiem quanh do.
    """
    stand = STAND if stand is None else stand
    return [stand[i] + action_scale * float(a[i]) for i in range(ACTION_DIM)]


# ====================================================================================
# 6) LUU / NAP CHECKPOINT
# ====================================================================================
def save_checkpoint(path, policy, meta):
    """Luu trong so + metadata (de --resume / --mode eval dung lai dung mang)."""
    torch.save({"state_dict": policy.state_dict(), "meta": dict(meta)}, path)


def load_checkpoint(path, device):
    """-> (meta, state_dict)."""
    ckpt = torch.load(path, map_location=device)
    return ckpt.get("meta", {}), ckpt["state_dict"]


# ====================================================================================
# 7) CHUONG TRINH HUAN LUYEN (TRAINING LOOP)
# ====================================================================================
def resolve_device(spec):
    """Chon thiet bi: --device cuda/cpu, mac dinh tu dong (cuda neu co)."""
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_policy(cfg, device):
    """
    Tao PolicyNet + nap checkpoint neu co --resume.
    -> (policy, stand, action_scale, meta)
    Luu y: kich thuoc mang (hidden) lay tu checkpoint de load duoc trong so cu.
    """
    meta, state_dict = {}, None
    if cfg.resume:
        meta, state_dict = load_checkpoint(cfg.resume, device)
        cfg.hidden = int(meta.get("hidden", cfg.hidden))     # checkpoint quyet dinh kien truc

    policy = PolicyNet(STATE_DIM, ACTION_DIM, cfg.hidden).to(device)
    if state_dict is not None:
        policy.load_state_dict(state_dict)

    stand = list(meta.get("stand", STAND))
    action_scale = float(meta.get("action_scale", cfg.action_scale))
    return policy, stand, action_scale, meta


def print_training_header(cfg, device, action_scale, ep_start):
    """In thong tin cau hinh truoc khi hoc (de log ro rang, tai lap duoc)."""
    log("=" * 104)
    log("HUAN LUYEN BO XUONG TU HOC: THANG BANG + DUNG THANG DUNG + VUON CAO")
    log("-" * 104)
    log("  Thiet bi   : %s | mang: 39 -> %d -> %d -> (mu:10 | V:1), tanh xuyen suot"
        % (device, cfg.hidden, cfg.hidden))
    log("  Algorithm  : %s%s"
        % ("A2C (cap nhat moi buoc)" if cfg.update == "online" else "REINFORCE theo episode",
           "" if not cfg.no_baseline else " + KHONG dung baseline"))
    log("  Optimizer  : Adam(lr=%g) | gamma=%g | entropy_coef=%g | grad_clip=%g"
        % (cfg.lr, cfg.gamma, cfg.entropy_coef, cfg.max_grad_norm))
    log("  Critic     : V(s) du doan return chiet khau; value loss duoc chia cho phuong sai target")
    log("  Action     : targetAngle[i] = STAND[i] + %g * tanh(policy)   (10 khop)" % action_scale)
    log("  Episode    : %s x toi da %d buoc (~%.1f giay mo phong/episode)"
        % (("khong gioi han" if cfg.episodes >= 10 ** 9 else "%d episode" % cfg.episodes),
           cfg.steps, cfg.steps * FRAME_DT))
    if cfg.time_seconds:
        log("  Thoi gian  : chay dung %s (ket thuc du kien luc %s) - het gio se tu dung va luu"
            % (hhmmss(cfg.time_seconds),
               time.strftime("%H:%M:%S", time.localtime(time.time() + cfg.time_seconds))))
    log("  Reward     : H=%.2f U=%.2f S=%.2f B=%.2f C=%.2f A=%.2f | fall: hip_y<%.2f -> %.1f diem"
        % (W_HEIGHT, W_UPRIGHT, W_STABILITY, W_BALANCE, W_CENTER, W_ALIVE,
           FALL_HEIGHT, FALL_PENALTY))
    log("  Do cao     : height_unit = clip((hip_y - %.3f)/(%.3f - %.3f), 0, 1)  (hip_y = hip.pos.y/600)"
        % (HEIGHT_FLOOR, HEIGHT_REF, HEIGHT_FLOOR))
    log("  Checkpoint : %s%s" % (cfg.save,
        ("   (resume tu '%s', tiep tuc tu episode %d)" % (cfg.resume, ep_start)) if cfg.resume else ""))
    log("=" * 104)


def save_state(cfg, policy, ckpt_meta, episode, best_height, reason=""):
    """Luu checkpoint kem metadata + backup '.bak' (chong mat du lieu khi ghi dung luc mat dien)."""
    ckpt_meta = dict(ckpt_meta, saved_episode=episode, best_mean_height=best_height)
    if cfg.save:                            # phien dai: giu luon ban truoc de phong file ghi hong
        try:
            if os.path.isfile(cfg.save):
                os.replace(cfg.save, cfg.save + ".bak")
        except OSError:
            pass                            # khong backup duoc thi van ghi binh thuong
    save_checkpoint(cfg.save, policy, ckpt_meta)
    log("  [SAVE] da luu '%s' (episode %d, best_mean_height=%.3f)%s"
        % (cfg.save, episode, best_height, (" - " + reason) if reason else ""))


def run_training(cfg):
    """Vong lap huan luyen: tuong tac env.exe -> thu thap (s, a, r, s') -> loss.backward() -> Adam."""
    require_torch()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    policy, stand, action_scale, meta = build_policy(cfg, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    ep_start = int(meta.get("saved_episode", -1)) + 1        # resume thi tiep tuc dem episode
    tgt_stat = RunningStat()                                 # phuong sai target critic (chuan hoa value loss)
    adv_stat = RunningStat()                                 # chuan hoa advantage truc tuyen
    w_reward, w_loss = Window(20), Window(20)
    w_steps, w_height = Window(20), Window(20)
    w_surv = Window(20)                                      # ti le episode song tron ven
    best_height = float(meta.get("best_mean_height", -1e9))

    ckpt_meta = {
        "stand": list(stand), "action_scale": action_scale, "hidden": cfg.hidden,
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
        "reward_weights": {"height": W_HEIGHT, "upright": W_UPRIGHT, "stability": W_STABILITY,
                           "balance": W_BALANCE, "center": W_CENTER, "alive": W_ALIVE,
                           "fall_penalty": FALL_PENALTY, "fall_height": FALL_HEIGHT,
                           "height_floor": HEIGHT_FLOOR, "height_ref": HEIGHT_REF,
                           "tilt_tol_deg": TILT_TOL_DEG},
        "best_mean_height": best_height,
    }
    print_training_header(cfg, device, action_scale, ep_start)

    total_updates = 0                                        # so lan loss.backward() da chay
    total_steps = 0                                          # tong so buoc vat ly (de tinh buoc/giay)
    t_start = time.time()                                    # moc bat dau phien (bao cao thoi gian chay)
    ep, last_info = ep_start - 1, None                       # ep = episode cuoi da chay xong

    # ---- Ngan sach thoi gian: chay "dung 1 gio" thay vi dem episode ----
    budget = TimeBudget(cfg.time_seconds) if cfg.time_seconds else None
    last_progress = time.time()
    time_up = False

    try:
        while True:
            # ---- Dieu kien dung: het so episode cho phep HOAC het gio ----
            if ep - ep_start + 1 >= cfg.episodes:
                log("[DUNG] da chay du %d episode theo --episodes." % cfg.episodes)
                break
            if budget is not None and budget.expired():
                log("[DUNG] da het thoi gian %s." % hhmmss(budget.total))
                break
            ep += 1

            # ---------------- Bat dau episode: reset bo xuong ve tu the goc ----------------
            reset_env()
            render_tick(not cfg.no_render)
            time.sleep(FRAME_DT)
            raw = request_state_retry(cfg.retry_seconds)
            x_home = raw[IDX_HIP_POS_X]              # moc X luc bat dau (thuong "dung yen 1 cho")
            height_start = _clamp01(raw[IDX_HIP_POS_Y])

            episode_reward, surv_steps = 0.0, 0
            max_height, last_info = height_start, None
            heights = []                                     # do cao tung buoc (de tinh TB)
            buf_logp, buf_value, buf_entropy, buf_reward = [], [], [], []   # cho che do episode

            for t in range(cfg.steps):
                # --------- 1. Policy chon action tu state hien tai ---------
                state_t = normalize_state(raw, device)
                action, log_prob, value, entropy = policy.act(state_t)
                send(*joint_target_from_action(action, stand, action_scale))

                # --------- 2. Tien 1 buoc vat ly trong env.exe ------------
                raw_next = request_state_retry(cfg.retry_seconds)
                render_tick(not cfg.no_render)

                # --------- 3. Tinh reward cho (s, a, s') ------------------
                reward, done, info = compute_reward(raw_next, x_home)
                episode_reward += reward
                surv_steps = t + 1
                max_height = max(max_height, info["hip_y"])
                last_info = info
                heights.append(info["hip_y"])

                # --------- 4. Tinh LOSS + LAN TRUYEN NGUOC + ADAM ---------
                loss_val = loss_pi_val = loss_v_val = ent_val = float("nan")
                if not cfg.no_baseline:
                    with torch.no_grad():
                        _, _, value_next = policy.forward(normalize_state(raw_next, device))
                    # KHONG nhan reward voi (1-gamma): lam vay se thu nho advantage that
                    # xuong ~0.01 - cung bac voi nhieu so cua critic -> ty le
                    # tin hieu/nhieu giam ~100 lan => policy bi "di loan" (Adam chuan hoa
                    # do lon gradient tung tham so nen nhieu cung bi khuech dai thanh buoc
                    # di that). Giu advantage tren thang do reward goc (O(1)) va chuan hoa
                    # rieng LOSS CUA CRITIC theo phuong sai truc tuyen cua target.
                    target = reward + (0.0 if done else cfg.gamma) * float(value_next)
                    adv = target - float(value.detach())
                    adv_stat.update(adv)
                    tgt_stat.update(target)
                    adv_t = torch.tensor((adv - adv_stat.mean) / adv_stat.std,
                                         dtype=torch.float32, device=device)
                    tgt_t = torch.tensor(target, dtype=torch.float32, device=device)
                    loss_pi = -(log_prob * adv_t)                         # policy gradient
                    # Chia value loss cho phuong sai truc tuyen cua target: V(s) phai hoc
                    # du doan return chiet khau ~ +100..+200, MSE tho se ~ 1e4 va lan at
                    # gradient cua policy tren trunk dung chung.
                    loss_v = F.mse_loss(value, tgt_t) / (tgt_stat.var + 1.0)   # critic MSE
                elif cfg.update == "online":
                    # REINFORCE toi gian (khong critic): dung reward tuc thoi lam trong so
                    r_t = torch.tensor(reward, dtype=torch.float32, device=device)
                    loss_pi = -(log_prob * r_t)
                    loss_v = torch.zeros((), device=device)
                else:
                    loss_pi = torch.zeros((), device=device)
                    loss_v = torch.zeros((), device=device)

                loss = loss_pi + 0.5 * loss_v - cfg.entropy_coef * entropy.mean()

                if cfg.update == "online":
                    optimizer.zero_grad()
                    loss.backward()                                       # <-- LAN TRUYEN NGUOC
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()                                      # <-- ADAM TOI UU TRONG SO
                    total_updates += 1
                    loss_val, loss_pi_val = float(loss.detach()), float(loss_pi.detach())
                    loss_v_val, ent_val = float(loss_v.detach()), float(entropy.mean().detach())
                else:
                    buf_logp.append(log_prob)
                    buf_value.append(value)
                    buf_entropy.append(entropy)
                    buf_reward.append(reward)

                # --------- 5. LOG tung buoc: Step Reward / Total Reward / Loss ---------
                if not cfg.quiet and t % cfg.log_every == 0:
                    extra = ("loss=%+.4f (pi=%+.4f v=%+.4f ent=%.3f)"
                             % (loss_val, loss_pi_val, loss_v_val, ent_val)
                             if cfg.update == "online" else "loss=(cap nhat cuoi episode)")
                    log("[B %6d] ep %03d | step %3d/%3d | r_step=%+.3f (%s) | R_total=%+.2f | %s | h=%.3f tilt=%2.0fdeg gnd=%d"
                        % (total_updates, ep, t + 1, cfg.steps, reward, fmt_reward(info),
                           episode_reward, extra, info["hip_y"], info["tilt_deg"], info["grounded"]))

                # --------- 6. Sang state tiep theo / ket thuc episode -------
                raw = raw_next
                total_steps += 1
                time.sleep(FRAME_DT)                 # giu nhip ~60Hz cho khop voi env.exe
                if done:
                    break
                if budget is not None and budget.expired():     # het gio giua episode
                    time_up = True
                    break

            # ---------------- Che do REINFORCE: cap nhat 1 lan cho ca episode ----------
            if cfg.update == "episode" and buf_reward:
                returns, g = [], 0.0
                for r in reversed(buf_reward):               # G_t = r_t + gamma * G_{t+1}
                    g = r + cfg.gamma * g
                    returns.append(g)
                returns.reverse()
                returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
                logp_t = torch.stack(buf_logp)
                value_t = torch.stack(buf_value)
                entropy_t = torch.stack(buf_entropy)

                # Chuan hoa return (zero-mean, unit-std) truoc khi dung lam target cho
                # critic + baseline. Return cong don ~ +100..+250 trong khi V(s) moi
                # khoi tao ~ 0 => MSE rat lon va lan at gradient cua policy.
                ret_mean, ret_std = returns_t.mean(), returns_t.std() + 1e-8
                returns_norm = (returns_t - ret_mean) / ret_std

                if cfg.no_baseline:
                    adv = returns_norm
                else:
                    adv = returns_norm - value_t.detach()    # baseline tru phuong sai
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                loss_pi = -(logp_t * adv).mean()             # REINFORCE loss
                loss_v = (torch.zeros((), device=device) if cfg.no_baseline
                          else F.mse_loss(value_t, returns_norm))
                loss = loss_pi + 0.5 * loss_v - cfg.entropy_coef * entropy_t.mean()

                optimizer.zero_grad()
                loss.backward()                              # <-- LAN TRUYEN NGUOC
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()                             # <-- ADAM TOI UU TRONG SO
                total_updates += 1
                loss_val, loss_pi_val = float(loss.detach()), float(loss_pi.detach())
                loss_v_val, ent_val = float(loss_v.detach()), float(entropy_t.mean().detach())
                log("  [UPDATE episode] loss=%+.4f (policy=%+.4f value=%+.4f entropy=%.3f) "
                    "| R_total=%+.2f | return TB=%+.2f | advantage TB=%+.3f"
                    % (loss_val, loss_pi_val, loss_v_val, ent_val, episode_reward,
                       float(returns_t.mean()), float(adv.mean())))

            # ---------------- NHAT KY EPISODE (Total Reward / Step Reward / Loss) -------
            mean_h = (sum(heights) / len(heights)) if heights else height_start
            ended_by_fall = bool(last_info and last_info["done"])
            prev_reward = w_reward.last if w_reward.buf else episode_reward
            w_reward.add(episode_reward)
            w_steps.add(surv_steps)
            w_height.add(mean_h)
            w_surv.add(1.0 if surv_steps >= cfg.steps else 0.0)
            w_loss.add(loss_val)
            trend = "UP" if episode_reward > prev_reward else ("DOWN" if episode_reward < prev_reward else "=")

            log("-" * 104)
            log("EPISODE %03d KET THUC: %s sau %d/%d buoc"
                % (ep, "NGA (hip tut duoi %.2f)" % FALL_HEIGHT if ended_by_fall else "HET THOI GIAN",
                   surv_steps, cfg.steps))
            log("  R_total   = %+.2f  (%s %+.2f so voi episode truoc) | Reward TB = %+.3f/buoc"
                % (episode_reward, trend, episode_reward - prev_reward,
                   episode_reward / max(surv_steps, 1)))
            log("  Reward cuoi: %s" % (fmt_reward(last_info) if last_info else "n/a"))
            log("  Do cao hip: bat dau %.3f -> cuoi %.3f | cao nhat %.3f | TB %.3f"
                % (height_start, last_info["hip_y"] if last_info else height_start, max_height, mean_h))
            log("  Loss      = %+.4f (policy %+.4f | value %+.4f | entropy %.3f) | tong backward = %d"
                % (loss_val, loss_pi_val, loss_v_val, ent_val, total_updates))
            log("  TB 20 ep  : R=%+.2f | buoc song=%.1f | do cao=%.3f | song tron %.0f%%"
                % (w_reward.mean, w_steps.mean, w_height.mean, 100.0 * w_surv.mean))
            log("-" * 104)

            # ---------------- Luu checkpoint dinh ky / khi lap ky luc ----------------
            new_best = w_height.mean > best_height and len(w_height.buf) >= 5
            if new_best:
                best_height = w_height.mean
            if cfg.save and (new_best or (cfg.save_every > 0 and (ep + 1) % cfg.save_every == 0)):
                save_state(cfg, policy, ckpt_meta, ep, best_height,
                           "ky luc do cao TB moi" if new_best else "dinh ky")

            # ---------------- Het gio giua episode -> dung ngay, khong doi het episode ----
            if time_up:
                log("[DUNG] het thoi gian giua episode %d (sau %d/%d buoc) - dang luu checkpoint..."
                    % (ep, surv_steps, cfg.steps))
                break

            # ---------------- Dong [TIEN DO] moi ~60 giay (cho phien dai nhu 1 gio) ----
            if budget is not None and time.time() - last_progress >= 60.0:
                last_progress = time.time()
                log(budget.progress_line(
                    "ep %d | backward %d | %.1f buoc/giay | TB 20ep: R=%+.1f, cao=%.3f, song tron %.0f%%"
                    % (ep, total_updates, total_steps / max(budget.elapsed, 1e-9),
                       w_reward.mean, w_height.mean, 100.0 * w_surv.mean)))

    except KeyboardInterrupt:
        log("")
        log("[STOP] Nguoi dung dung huan luyen bang Ctrl+C - dang luu checkpoint...")
    except EnvNotRunning as exc:
        log("[LOI] %s" % exc)
    finally:
        if cfg.save:
            save_state(cfg, policy, ckpt_meta, ep, best_height, "ket thuc phien huan luyen")
        elapsed = time.time() - t_start
        log("TONG KET PHIEN: chay %s | %d episode | %d buoc vat ly | %.1f buoc/giay | backward = %d"
            % (hhmmss(elapsed), max(0, ep - ep_start + 1), total_steps,
               total_steps / max(elapsed, 1e-9), total_updates))
        if cfg.time_seconds:
            log("Ngan sach thoi gian: %s | da dung: %s | con lai: %s"
                % (hhmmss(cfg.time_seconds), hhmmss(elapsed),
                   hhmmss(max(0.0, cfg.time_seconds - elapsed))))
        log("Xem lai policy da hoc:  python src/env_bridge.py --mode eval --resume %s" % cfg.save)
        log("=" * 104)


# ====================================================================================
# 8) CHE DO XEM THU (EVAL) - dung policy da hoc, KHONG cap nhat trong so
# ====================================================================================
def run_eval(cfg):
    """Chay policy o che do deterministic (a = tanh(mu)) de danh gia ky nang da hoc."""
    require_torch()
    device = resolve_device(cfg.device)
    policy, stand, action_scale, meta = build_policy(cfg, device)
    policy.eval()

    log("=" * 104)
    log("XEM THU POLICY: %s | thiet bi=%s | action_scale=%g" % (cfg.resume or "(mang moi, chua hoc)", device, action_scale))
    if cfg.time_seconds:
        log("Thoi gian  : chay dung %s (ket thuc du kien luc %s)"
            % (hhmmss(cfg.time_seconds),
               time.strftime("%H:%M:%S", time.localtime(time.time() + cfg.time_seconds))))
    log("=" * 104)

    # ---- Ngan sach thoi gian: chay xem mo hinh trong 1 gio chang han ----
    budget = TimeBudget(cfg.time_seconds) if cfg.time_seconds else None
    last_progress = time.time()
    total_steps = 0
    t_start = time.time()
    ep, time_up = -1, False
    all_rewards, all_heights, all_steps = [], [], []
    try:
        while True:
            if ep + 1 >= cfg.episodes:                  # het so episode cho phep
                break
            if budget is not None and budget.expired():  # het gio
                log("[DUNG] da het thoi gian %s." % hhmmss(budget.total))
                break
            ep += 1
            reset_env()
            render_tick(not cfg.no_render)
            time.sleep(FRAME_DT)
            raw = request_state_retry(cfg.retry_seconds)
            x_home = raw[IDX_HIP_POS_X]
            ep_reward, surv = 0.0, 0
            heights, last_info = [], None

            for t in range(cfg.steps):
                state_t = normalize_state(raw, device)
                with torch.no_grad():
                    action, _, _, _ = policy.act(state_t, deterministic=True)
                send(*joint_target_from_action(action, stand, action_scale))
                raw_next = request_state_retry(cfg.retry_seconds)
                render_tick(not cfg.no_render)
                reward, done, info = compute_reward(raw_next, x_home)
                ep_reward += reward
                surv = t + 1
                heights.append(info["hip_y"])
                last_info = info
                raw = raw_next
                if not cfg.quiet and t % cfg.log_every == 0:
                    log("  ep %03d step %3d/%3d | r_step=%+.3f (%s) | R=%+.2f | h=%.3f"
                        % (ep, t + 1, cfg.steps, reward, fmt_reward(info), ep_reward, info["hip_y"]))
                time.sleep(FRAME_DT)
                total_steps += 1
                if done:
                    break
                if budget is not None and budget.expired():     # het gio giua episode
                    time_up = True
                    break

            all_rewards.append(ep_reward)
            all_steps.append(surv)
            all_heights.append(sum(heights) / max(len(heights), 1))
            log("[EVAL ep %03d] R_total=%+.2f | %s | buoc=%d | do cao TB=%.3f | cao nhat=%.3f"
                % (ep, ep_reward, "NGA" if (last_info and last_info["done"]) else "HET GIO",
                   surv, all_heights[-1], max(heights) if heights else 0.0))

            if time_up:                                  # het gio giua episode -> dung
                log("[DUNG] het thoi gian giua episode %d (sau %d/%d buoc)." % (ep, surv, cfg.steps))
                break

            # ---------------- Dong [TIEN DO] moi ~60 giay (phien dai nhu 1 gio) --------
            if budget is not None and time.time() - last_progress >= 60.0:
                last_progress = time.time()
                log(budget.progress_line(
                    "ep %d | %d buoc | %.1f buoc/giay | R TB=%+.1f | cao TB=%.3f"
                    % (ep + 1, total_steps, total_steps / max(budget.elapsed, 1e-9),
                       sum(all_rewards) / max(len(all_rewards), 1),
                       sum(all_heights) / max(len(all_heights), 1))))
    except KeyboardInterrupt:
        log("[STOP] dung xem thu.")
    except EnvNotRunning as exc:
        log("[LOI] %s" % exc)
    finally:
        if all_rewards:
            log("-" * 104)
            log("TONG KET %d episode: R TB=%+.2f | buoc TB=%.1f | do cao TB=%.3f"
                % (len(all_rewards), sum(all_rewards) / len(all_rewards),
                   sum(all_steps) / len(all_steps), sum(all_heights) / len(all_heights)))
        elapsed = time.time() - t_start
        log("Thoi gian da chay: %s | %d buoc vat ly | %.1f buoc/giay"
            % (hhmmss(elapsed), total_steps, total_steps / max(elapsed, 1e-9)))
        if cfg.time_seconds:
            log("Ngan sach thoi gian: %s | da dung: %s | con lai: %s"
                % (hhmmss(cfg.time_seconds), hhmmss(elapsed),
                   hhmmss(max(0.0, cfg.time_seconds - elapsed))))
        log("=" * 104)


# ====================================================================================
# 8b) HAM CHAY THEO THOI GIAN - "chay mo hinh trong vong 1 gio"
# ====================================================================================
def run_timed(cfg, seconds=3600.0):
    """
    Chay mo hinh trong dung `seconds` GIAY (thoi gian THUC, wall-clock) roi tu dung
    va luu checkpoint. Day la ham ban yeu cau: chay 1 gio.

        seconds = 3600  -> 1 gio (mac dinh)
        seconds = 1800  -> 30 phut
        seconds = 60    -> 1 phut (de thu nhanh xem co dung han khong)

    Hanh vi theo cfg.mode:
        'train' -> HUAN LUYEN trong 1 gio (A2C cap nhat tung buoc + loss.backward())
        'eval'  -> CHAY/TRINH DIEN mo hinh da hoc lien tuc trong 1 gio (khong cap nhat)

    Luu y:
      * Het gio se dung NGAY (ke ca giua episode) va LUON luu checkpoint truoc khi thoat.
      * Neu truyen ca --episodes thi do la GIOI HAN CUNG thu hai (dung o cai nao toi truoc).
      * Tu dong lenh: --time 1h / 30m / 90s / 45 (=45 phut).
    """
    cfg.time_seconds = float(seconds)
    if cfg.mode not in ("train", "eval"):
        log("[LUU Y] --time chi co y nghia voi --mode train / --mode eval (dang la '%s')."
            % cfg.mode)
        return
    log("[THOI GIAN] mode '%s' se chay trong %s, ket thuc du kien luc %s."
        % (cfg.mode, hhmmss(seconds),
           time.strftime("%H:%M:%S", time.localtime(time.time() + seconds))))
    return run_training(cfg) if cfg.mode == "train" else run_eval(cfg)


def run_one_hour(cfg):
    """Tien dung cho dung yeu cau 'chay trong 1h' (goi y them --no-render cho nhanh)."""
    return run_timed(cfg, 3600.0)


def run_ten_hours(cfg):
    """
    Tien dung: chay LIEN TUC 10 GIO (36.000 giay) khong nghi.
        python src/env_bridge.py --time 10h  (hoac goi ham nay trong code)
    An toan cho phien dai:
      - tu dong luu checkpoint moi --save-every episode + ban backup '.bak';
      - tu dong thu lai khi env.exe mat ket noi tam thoi trong han --retry (mac dinh 120s);
      - Ctrl+C cung luu checkpoint truoc khi thoat;
      - may bi tat giua chung? mo lai env.exe va hoc tiep bang --resume rha_policy.pt.
    """
    return run_timed(cfg, 10 * 3600.0)


# ====================================================================================
# 9) CHE DO CU (TUONG THICH NGUOC): di bo bang dao dong sin / dung yen tai cho
# ====================================================================================
GAIT_FREQ = 1.5      # Hz - tan so buoc chan giay
SWING_LEG = 0.35     # rad - bien do vung dui
SWING_KNEE = 0.45    # rad - bien do gap goi
SWING_ARM = 0.25     # rad - bien do vung tay

# Bien do dao dong cho tung khop (0 = giu nguyen tu the STAND)
AMP = [0.0,          # head
       0.0,          # hip
       SWING_ARM,    # armR  - vung nguoc voi chan ben phai
       SWING_ARM,    # armL  - vung nguoc voi chan ben trai
       0.0,          # forearmR
       0.0,          # forearmL
       SWING_LEG,    # legR
       SWING_LEG,    # legL  - vung nguoc pha voi legR
       SWING_KNEE,   # calfR - gap goi pha lech
       SWING_KNEE]   # calfL

PHASE = [0.0,           # head
         0.0,           # hip
         math.pi,       # armR
         0.0,           # armL
         0.0, 0.0,      # forearmR, forearmL
         0.0,           # legR
         math.pi,       # legL
         math.pi / 2,   # calfR
         3 * math.pi / 2]   # calfL


def run_walk(cfg):
    """Che do cu: action[i] = STAND[i] + AMP[i]*amp*sin(2*pi*f*t + PHASE[i])."""
    limit = cfg.frames
    frame, amp_scale = 0, cfg.amp
    budget = TimeBudget(cfg.time_seconds) if cfg.time_seconds else None
    log("Che do WALK: freq=%gHz amp=%g (Ctrl+C de dung)%s"
        % (cfg.freq, cfg.amp, "" if limit is None else " | toi da %d frame" % limit))
    while (limit is None or frame < limit) and not (budget is not None and budget.expired()):
        state = request_state()
        t = frame * FRAME_DT
        w = 2.0 * math.pi * cfg.freq
        action = [STAND[i] + AMP[i] * amp_scale * math.sin(w * t + PHASE[i])
                  for i in range(ACTION_DIM)]
        send(*action)
        if not cfg.quiet and frame % 15 == 0:
            vals = " ".join("%s=%+.2f" % (n, a) for n, a in zip(JOINT_NAMES, action))
            log("frame %04d | hip=(%.0f,%.0f) | %s"
                % (frame, state[IDX_HIP_POS_X] * 800, state[IDX_HIP_POS_Y] * 600, vals))
        render_tick(not cfg.no_render)
        time.sleep(FRAME_DT)
        frame += 1
    log("Hoan thanh %d frame state<->action" % frame)


def run_stand(cfg):
    """Che do cu: giu nguyen tu the STAND (khong dao dong)."""
    limit = cfg.frames
    frame = 0
    budget = TimeBudget(cfg.time_seconds) if cfg.time_seconds else None
    log("Che do STAND: giu tu the dung thang (Ctrl+C de dung)%s"
        % ("" if limit is None else " | toi da %d frame" % limit))
    while (limit is None or frame < limit) and not (budget is not None and budget.expired()):
        state = request_state()
        send(*STAND)
        if not cfg.quiet and frame % 15 == 0:
            log("frame %04d | hip=(%.0f,%.0f) | grounded L/R=%d/%d"
                % (frame, state[IDX_HIP_POS_X] * 800, state[IDX_HIP_POS_Y] * 600,
                   int(state[IDX_GROUND_L]), int(state[IDX_GROUND_R])))
        render_tick(not cfg.no_render)
        time.sleep(FRAME_DT)
        frame += 1
    log("Hoan thanh %d frame state<->action" % frame)


# ====================================================================================
# 10) THAM SO DONG LENH + MAIN
# ====================================================================================
EPILOG = """\
Vi du:
  # 0) Mo env.exe truoc (no la server UDP: nhan lenh o port 5005, gui state ve 5006)
  # 1) HUAN LUYEN DUNG 1 GIO - het gio tu dong dung va luu checkpoint
  python src/env_bridge.py --mode train --time 1h --no-render --save-every 25 --log-every 100
  # 1b) CHAY LIEN TUC 10 GIO KHONG NGHI (mo env.exe + nhan phim Y truoc khi chay)
  python src/env_bridge.py --mode train --time 10h --no-render --save-every 20 --log-every 200
  # 1c) May bi tat giua chung? Mo lai env.exe roi hoc TIEP 10 gio nua
  python src/env_bridge.py --mode train --resume rha_policy.pt --time 10h --no-render
  # 2) Chay / trinh dien mo hinh da hoc lien tuc 1 gio
  python src/env_bridge.py --mode eval --resume rha_policy.pt --time 1h --no-render
  # 3) Huan luyen theo so episode (A2C, cap nhat moi buoc)
  python src/env_bridge.py --mode train --episodes 300 --steps 300 --lr 3e-4
  # 4) Huan luyen kieu REINFORCE theo episode (loss.backward() 1 lan/episode)
  python src/env_bridge.py --mode train --update episode --episodes 300 --steps 300
  # 5) Hoc tiep tu checkpoint cu (van gioi han 1h neu truyen --time)
  python src/env_bridge.py --mode train --resume rha_policy.pt --time 1h
  # 6) Xem policy da hoc (greedy, khong hoc)
  python src/env_bridge.py --mode eval --resume rha_policy.pt --steps 600 --episodes 3
  # 7) Tang toc: --no-render + nhan phim Y trong cua so env.exe (frameInterval=8)
  python src/env_bridge.py --mode train --no-render --episodes 300
  # 8) Che do cu: di bo bang dao dong sin / dung yen
  python src/env_bridge.py --mode walk --freq 1.5 --amp 1.0 --frames 600
  python src/env_bridge.py --mode stand --frames 600
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="env_bridge.py",
        description="Cau noi UDP (Python <-> env.exe) + huan luyen bo xuong tu hoc bang PyTorch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)

    # ---- Che do chay ----
    p.add_argument("--mode", choices=["train", "eval", "walk", "stand"], default="train",
                   help="train = huan luyen (mac dinh) | eval = chay thu policy | "
                        "walk/stand = che do cu")
    # ---- Huan luyen ----
    p.add_argument("--episodes", type=int, default=None,
                   help="so episode (mac dinh 300; neu dung --time ma khong dat --episodes thi "
                        "KHONG gioi han theo episode, chi dung theo thoi gian)")
    p.add_argument("--time", default=None, metavar="DURATION",
                   help="CHAY TRONG BAO LAU: 1h, 30m, 90s, 45 (=45 phut). Het gio se TU DUNG va "
                        "luu checkpoint (dung ca giua episode). Ap dung cho moi --mode")
    p.add_argument("--steps", type=int, default=300,
                   help="so buoc toi da moi episode, ~1/60 giay/buoc (mac dinh 300 = 5 giay)")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate cua Adam (mac dinh 3e-4)")
    p.add_argument("--gamma", type=float, default=0.995,
                   help="he so chiet khau (mac dinh 0.995 ~ tam nhin 200 buoc = 3.3 giay; "
                        "gamma thap se lam hinh phat NGA o cuoi episode bi chiet khau qua nho)")
    p.add_argument("--hidden", type=int, default=128, help="so neuron moi lop an (mac dinh 128)")
    p.add_argument("--action-scale", type=float, default=0.35,
                   help="bien do lech quanh STAND, rad: action = STAND + scale*tanh(policy)")
    p.add_argument("--update", choices=["online", "episode"], default="online",
                   help="online = A2C cap nhat tung buoc | episode = REINFORCE cap nhat cuoi episode")
    p.add_argument("--entropy-coef", type=float, default=0.002,
                   help="he so entropy (kham pha). log_std la tham so dung chung nen he so nay "
                        "phai NHO, neu khong sigma se bi day len tran va policy khong hoi tu")
    p.add_argument("--no-baseline", action="store_true",
                   help="tat critic baseline (REINFORCE thuan, phuong sai cao)")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="clip gradient (mac dinh 1.0)")
    p.add_argument("--seed", type=int, default=1234, help="seed cho PyTorch")
    p.add_argument("--device", default=None, help="cpu | cuda (mac dinh tu dong)")
    p.add_argument("--save", default="rha_policy.pt", help="duong dan checkpoint ('' = khong luu)")
    p.add_argument("--save-every", type=int, default=20, help="luu checkpoint moi N episode")
    p.add_argument("--resume", default=None, help="nap checkpoint truoc khi huan luyen / eval")
    p.add_argument("--log-every", type=int, default=25, help="in log moi N buoc (mac dinh 25)")
    p.add_argument("--no-render", action="store_true",
                   help="khong yeu cau env.exe ve frame (nhanh hon neu vong lap env khong bi vsync)")
    p.add_argument("--quiet", action="store_true", help="chi in tong ket episode, bo log tung buoc")
    p.add_argument("--retry", type=float, default=None,
                   help="phien dai (vi du 10h): so GIAY thu lai khi mat ket noi env.exe tam thoi "
                        "truoc khi thoat (mac dinh: 120 neu --time >= 1h, nguoc lai 15)")
    # ---- Che do cu (walk/stand) ----
    p.add_argument("--frames", type=int, default=None, help="[walk/stand] so frame toi da, bo trong = vo han")
    p.add_argument("--freq", type=float, default=GAIT_FREQ, help="[walk] tan so buoc (Hz)")
    p.add_argument("--amp", type=float, default=1.0, help="[walk] he so bien do dao dong")

    cfg = p.parse_args(argv)
    cfg.steps = max(1, int(cfg.steps))          # tranh vong lap rong
    cfg.log_every = max(1, int(cfg.log_every))

    # ---- Xu ly --time (vi du '1h') va gioi han episode ----
    cfg.time_seconds = parse_duration(cfg.time) if cfg.time is not None else None
    if cfg.episodes is None:
        # Co --time ma khong dat --episodes => chi dung theo THOI GIAN (khong gioi han episode)
        cfg.episodes = 10 ** 9 if cfg.time_seconds else 300
    else:
        cfg.episodes = max(1, int(cfg.episodes))    # nguoi dung dat => gioi han cung thu hai

    # ---- Han retry cho phien dai: mat ket noi tam thoi khong duoc lam chet phien ----
    cfg.retry_seconds = (120.0 if (cfg.time_seconds or 0.0) >= 3600.0 else 15.0
                         if cfg.retry is None else max(0.0, float(cfg.retry)))
    return cfg


def main(argv=None):
    cfg = parse_args(argv)

    # Bind socket nhan state (127.0.0.1:5006)
    try:
        open_sockets()
    except OSError as exc:
        log("[LOI] Khong bind duoc UDP port %d: %s" % (PY_STATE_ADDR[1], exc))
        log("      Co the mot tien trinh Python khac dang chay va giu port nay.")
        return 1

    rc = 0
    try:
        set_impulse(0.0)        # tat luc day ngau nhien cho bai toan hoc thang bang
        if cfg.mode in ("train", "eval"):
            if cfg.time_seconds:            # --time 1h => chay dung 1 gio roi tu dung
                run_timed(cfg, cfg.time_seconds)
            elif cfg.mode == "train":
                run_training(cfg)
            else:
                run_eval(cfg)
        elif cfg.mode == "walk":
            run_walk(cfg)
        else:
            run_stand(cfg)
    except KeyboardInterrupt:
        log("[STOP] nguoi dung dung chuong trinh.")
    except EnvNotRunning as exc:
        log("[LOI] %s" % exc)
        rc = 2
    except OSError as exc:
        log("[LOI] Loi socket UDP: %s" % exc)
        log("      Hay chac chan env.exe dang chay va khong co chuong trinh nao khac dung port 5006.")
        rc = 1
    finally:
        close_sockets()
    return rc


if __name__ == "__main__":
    sys.exit(main())