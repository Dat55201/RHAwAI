"""
env_test.py - CHUONG TRINH TEST / KIEM TRA MO HINH BO XUONG DA HOC
==================================================================
Doc so lieu da thu duoc trong qua trinh huan luyen (checkpoint + file log)
va "soi" xem bo xuong dang hoc duoc gi, ma KHONG can huan luyen lai.

4 phan:
  PHAN 1 - CHECKPOINT      : kien truc mang, tham so, sigma kham pha, cau hinh
                             reward, hanh dong policy tren tu the dung gia dinh.
  PHAN 2 - LOG HUAN LUYEN  : doc file log (vi du t1.log) -> bang theo episode,
                             xu the tien bo (dau vs cuoi), bieu do text, ti le nga.
  PHAN 3 - PHAN TICH OFFLINE: do nhay policy theo do cao (hip_y) va goc nghieng
                             (body_sin), do do lon hanh dong tren state nhieu.
  PHAN 4 - LIVE EVAL (tuy chon --live): chay that tren env.exe o che do
                             deterministic, xuat CSV tung buoc + thong ke episode.

Cach dung:
    python src/env_test.py                          # PHAN 1 + 2 + 3 (offline)
    python src/env_test.py --checkpoint rha_policy.pt --log t1.log
    python src/env_test.py --detail                 # them thong ke trong so tung lop
    python src/env_test.py --live                   # + PHAN 4 (can mo env.exe truoc)
    python src/env_test.py --live --episodes 3 --steps 150 --csv test_results/eval.csv

Ket qua cham diem (reward moi buoc / 2.0 diem toi da):
    TOT          : >= 1.5
    TRUNG BINH   : 1.0 .. 1.5
    KEM          : < 1.0
"""
import argparse
import csv
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_bridge as eb

if eb.TORCH_AVAILABLE:
    import torch
else:
    raise SystemExit(eb.TORCH_HINT)

SEP = "=" * 100
SUB = "-" * 100
MAX_STEP_REWARD = eb.W_HEIGHT + eb.W_UPRIGHT + eb.W_BALANCE + eb.W_ALIVE   # = 2.0


# ====================================================================================
# TIEN ICH
# ====================================================================================
def resolve_path(p):
    """Tim file: thu tu tren cung -> thu muc goc repo (cha cua src/)."""
    if os.path.isfile(p):
        return p
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(root, p)
    return cand if os.path.isfile(cand) else p


def read_text_any(path):
    """Doc file log bao moi encoding (PowerShell redirect 5.1 tao file UTF-16 LE)."""
    data = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("latin-1", errors="replace")


def classify(rate):
    """Cham diem theo reward moi buoc (toi da 2.0)."""
    if rate >= 1.5:
        return "TOT (>=1.5)"
    if rate >= 1.0:
        return "TRUNG BINH (1.0..1.5)"
    return "KEM (<1.0)"


def ascii_chart(values, width=46):
    """Bieu do text ngang: moi dong 1 gia tri, bar '#' theo ty le."""
    if not values:
        return "  (khong co du lieu)"
    lo, hi = min(values), max(values)
    rng = hi - lo
    lines = []
    for i, v in enumerate(values):
        if rng <= 1e-12:
            bar = "#" * (width // 2)
        else:
            bar = "#" * max(1, int(round((v - lo) / rng * (width - 1))) + 1)
        lines.append("  [%3d] %10.3f |%s" % (i, v, bar))
    return "\n".join(lines)


def policy_eval(policy, raw_state, stand, action_scale, device="cpu"):
    """
    Chay policy deterministic tren 1 state tho (39 float).
    -> (action_list_10, target_list_10, value, reward, done, info)
    """
    t = eb.normalize_state(raw_state, device).unsqueeze(0)
    with torch.no_grad():
        mu, _, value = policy.forward(t)
        a = torch.tanh(mu)[0]
    target = eb.joint_target_from_action(a, stand, action_scale)
    reward, done, info = eb.compute_reward(raw_state)
    return ([float(x) for x in a], target, float(value[0]),
            reward, done, info)


def make_home_state(hip_y=0.17):
    """
    State "tu the dung gia dinh" (39 float) dung cho phan tich offline:
    moi xuong sin=1/cos=0 (goc 0 = thang), hip_y = do cao chuan, hip_x = giua,
    khong van toc, ca 2 chan cham dat. Day khong phai state that cua env.exe
    (state that phu thuoc mo phong) nhung du de do nhay policy.
    """
    s = [0.0] * eb.STATE_DIM
    for i in range(len(eb.BONE_NAMES)):          # sin(angle) = 1 -> tilt = 0 do
        s[i * 3] = 1.0
    s[eb.IDX_HIP_POS_Y] = hip_y
    s[eb.IDX_HIP_POS_X] = 0.5
    s[eb.IDX_HIP_VEL_Y] = 0.0
    s[eb.IDX_HIP_VEL_X] = 0.0
    s[eb.IDX_GROUND_L] = 1.0
    s[eb.IDX_GROUND_R] = 1.0
    return s


# ====================================================================================
# PHAN 1 - KIEM TRA CHECKPOINT
# ====================================================================================
def load_policy(path):
    """Nap checkpoint -> (policy, meta, stand, action_scale)."""
    meta, sd = eb.load_checkpoint(path, "cpu")
    state_dim = int(meta.get("state_dim", eb.STATE_DIM))
    action_dim = int(meta.get("action_dim", eb.ACTION_DIM))
    hidden = int(meta.get("hidden", 128))
    policy = eb.PolicyNet(state_dim, action_dim, hidden)
    policy.load_state_dict(sd)
    policy.eval()
    stand = list(meta.get("stand", eb.STAND))
    action_scale = float(meta.get("action_scale", 0.35))
    return policy, meta, stand, action_scale


def part1_checkpoint(path, detail=False):
    print(SEP)
    print("PHAN 1: KIEM TRA CHECKPOINT: %s" % path)
    print(SEP)
    policy, meta, stand, action_scale = load_policy(path)
    sd = policy.state_dict()

    print("  Kich thuoc file     : %s bytes" % "{:,}".format(os.path.getsize(path)))
    print("  Kien truc mang      : %d -> %d -> %d -> (mu:%d | V:1) | tanh xuyen suot"
          % (int(meta.get("state_dim", 39)), int(meta.get("hidden", 128)),
             int(meta.get("hidden", 128)), int(meta.get("action_dim", 10))))
    print("  Tong tham so        : %s" % "{:,}".format(sum(v.numel() for v in sd.values())))
    print("  Action              : targetAngle[i] = STAND[i] + %.2f * tanh(policy)" % action_scale)
    print("  STAND               : %s" % [round(float(v), 2) for v in stand])
    print("  Luu tai episode     : %s" % meta.get("saved_episode"))
    bmh = meta.get("best_mean_height")
    if bmh is not None and bmh > -1e8:
        print("  best_mean_height    : %.4f  (tu the STAND goc ~ 0.169)" % bmh)
    else:
        print("  best_mean_height    : (chua co du lieu - checkpoint tu phien test ngan)")

    # --- do kham pha (sigma) ---------------------------------------------------------
    log_std = sd["log_std"]
    sigma = log_std.exp()
    print("")
    print("  DO KHAM PHA (log_std la tham so hoc duoc):")
    print("    log_std : mean=%.3f  min=%.3f  max=%.3f"
          % (float(log_std.mean()), float(log_std.min()), float(log_std.max())))
    print("    sigma   : mean=%.3f rad  (do lech chuan policy quanh mu, tinh theo moi khop)"
          % float(sigma.mean()))

    # --- cau hinh reward -------------------------------------------------------------
    rw = meta.get("reward_weights", {}) or {}
    print("")
    print("  CAU HINH REWARD (trong so luu trong checkpoint):")
    keys = [("height", "thuong do cao hip_y", eb.W_HEIGHT),
            ("upright", "thuong song lung thang dung", eb.W_UPRIGHT),
            ("stability", "phat rung lac / truot", eb.W_STABILITY),
            ("balance", "thuong 2 chan cham dat", eb.W_BALANCE),
            ("center", "phat roi khoi cho", eb.W_CENTER),
            ("alive", "thuong song sot", eb.W_ALIVE),
            ("fall_penalty", "phat nga", eb.FALL_PENALTY),
            ("fall_height", "nguong nga (hip_y)", eb.FALL_HEIGHT),
            ("height_floor", "san do cao", eb.HEIGHT_FLOOR),
            ("height_ref", "tran do cao (bao hoa)", eb.HEIGHT_REF),
            ("tilt_tol_deg", "dung sai nghieng (do)", eb.TILT_TOL_DEG)]
    for k, desc, dflt in keys:
        print("    %-14s = %-8s (%s)" % (k, rw.get(k, dflt), desc))
    print("    => Reward toi da 1 buoc (dung ngay, khong rung): %.2f" % MAX_STEP_REWARD)

    # --- hanh dong tren tu the dung gia dinh ----------------------------------------
    home = make_home_state()
    a, target, value, reward, done, info = policy_eval(policy, home, stand, action_scale)
    print("")
    print("  POLICY TREN TU THE DUNG GIA DINH (hip_y=0.17, song lung thang, 2 chan cham dat):")
    print("    tanh(policy)      : %s" % ["%+.3f" % v for v in a])
    print("    targetAngle (rad) : %s" % ["%+.2f" % v for v in target])
    print("    |a| trung binh    : %.4f  (0 = giu nguyen tu the STAND)" %
          (sum(abs(v) for v in a) / len(a)))
    print("    V(s) (critic)     : %.3f" % value)
    print("    reward state nay  : %+.3f  [%s]" % (reward, eb.fmt_reward(info)))

    # --- thong ke trong so (tuy chon --detail) --------------------------------------
    if detail:
        print("")
        print("  THONG KE TRONG SO TUNG LOP:")
        print("    %-18s %8s %8s %8s %8s" % ("layer", "mean", "std", "min", "abs_max"))
        for k, v in sorted(sd.items()):
            t = v.detach()
            if t.dim() >= 2:
                print("    %-18s %8.4f %8.4f %8.4f %8.4f"
                      % (k, float(t.mean()), float(t.std()), float(t.min()),
                         float(t.abs().max())))
    print(SEP)
    return policy, meta, stand, action_scale


# ====================================================================================
# PHAN 2 - PHAN TICH FILE LOG HUAN LUYEN
# ====================================================================================
RE_EPISODE = re.compile(r"EPISODE\s+(\d+)\s+KET THUC:\s*(.+?)\s+sau\s+(\d+)/(\d+)\s+buoc")
RE_RTOTAL = re.compile(r"R_total\s*=\s*([+-]?[\d.]+)\s*\(.*\)\s*\|\s*Reward TB\s*=\s*([+-]?[\d.]+)/buoc")
RE_COMP = re.compile(r"Reward cuoi:\s*H=([\d.]+)\s+U=([\d.]+)\s+S=([+-]?[\d.]+)\s+"
                     r"B=([+-]?[\d.]+)\s+C=([+-]?[\d.]+)\s+A=([\d.]+)")
RE_HEIGHT = re.compile(r"Do cao hip:\s*bat dau\s*([\d.]+)\s*->\s*cuoi\s*([\d.]+)\s*\|"
                       r"\s*cao nhat\s*([\d.]+)\s*\|\s*TB\s*([\d.]+)")
RE_LOSS = re.compile(r"Loss\s*=\s*([+-]?[\d.]+)\s*\(policy\s*([+-]?[\d.]+)\s*\|\s*value"
                     r"\s*([+-]?[\d.]+)\s*\|\s*entropy\s*([\d.]+)\)")
RE_TB20 = re.compile(r"TB 20 ep\s*:\s*R=([+-]?[\d.]+)\s*\|\s*buoc song=([\d.]+)\s*\|"
                     r"\s*do cao=([\d.]+)\s*\|\s*song tron\s*([\d.]+)%")
RE_SESSION = re.compile(r"Checkpoint\s*:\s*(\S+)")
RE_SUMMARY = re.compile(r"TONG KET PHIEN:\s*chay\s*(\S+)\s*\|\s*(\d+)\s*episode\s*\|\s*(\d+)"
                        r"\s*buoc vat ly\s*\|\s*([\d.]+)\s*buoc/giay")


def parse_log(path):
    """
    Doc file log huan luyen -> danh sach phien (session), moi phien chua cac episode.
    Phien moi bat dau khi gap dong 'Checkpoint : ...' trong header.
    """
    text = read_text_any(path)
    sessions = []                       # [{checkpoint, episodes:[...], summary}]
    cur = {"checkpoint": "?", "episodes": [], "summary": None}
    ep = None
    for line in text.splitlines():
        m = RE_SESSION.search(line)
        if m:
            if cur["episodes"]:
                sessions.append(cur)
            cur = {"checkpoint": m.group(1), "episodes": [], "summary": None}
            ep = None
            continue
        m = RE_EPISODE.search(line)
        if m:
            ep = {"ep": int(m.group(1)), "reason": m.group(2).strip(),
                  "step": int(m.group(3)), "step_max": int(m.group(4))}
            cur["episodes"].append(ep)
            continue
        if ep is None:
            continue
        m = RE_RTOTAL.search(line)
        if m:
            ep["R"] = float(m.group(1)); ep["R_rate"] = float(m.group(2)); continue
        m = RE_COMP.search(line)
        if m:
            (ep["H"], ep["U"], ep["S"], ep["B"], ep["C"], ep["A"]) = \
                [float(g) for g in m.groups()]
            continue
        m = RE_HEIGHT.search(line)
        if m:
            (ep["h0"], ep["h1"], ep["hmax"], ep["hmean"]) = \
                [float(g) for g in m.groups()]
            continue
        m = RE_LOSS.search(line)
        if m:
            (ep["loss"], ep["loss_pi"], ep["loss_v"], ep["entropy"]) = \
                [float(g) for g in m.groups()]
            continue
        m = RE_TB20.search(line)
        if m:
            ep["tb20_R"], ep["tb20_surv"], ep["tb20_h"], ep["tb20_pct"] = \
                [float(g) for g in m.groups()]
            continue
    if cur["episodes"]:
        sessions.append(cur)
    # Gan tung dong 'TONG KET PHIEN' cho dung phien theo thu tu (search() chi lay 1).
    summaries = [m.groups() for m in RE_SUMMARY.finditer(text)]
    for i, s in enumerate(sessions):
        if i < len(summaries):
            s["summary"] = summaries[i]
    return sessions


def part2_log(path):
    print("")
    print(SEP)
    print("PHAN 2: PHAN TICH FILE LOG HUAN LUYEN: %s" % path)
    print(SEP)
    if not os.path.isfile(path):
        print("  [BO QUA] khong tim thay file log: %s" % path)
        return
    sessions = parse_log(path)
    if not sessions or not any(s["episodes"] for s in sessions):
        print("  [BO QUA] khong doc duoc episode nao tu log (dinh dang khong khop?)")
        return
    eps_all = []
    for si, s in enumerate(sessions):
        eps = s["episodes"]
        eps_all.extend(eps)
        print("")
        print("  PHIEN %d - checkpoint: %s - %d episode" % (si + 1, s["checkpoint"], len(eps)))
        print("    %-6s %-18s %9s %10s %8s %8s %9s %8s" %
              ("ep", "ket thuc", "R_total", "R/buoc", "h_TB", "h_max", "loss", "entropy"))
        for e in eps:
            print("    %-6d %-18s %+9.2f %10.3f %8s %8s %9s %8s" %
                  (e["ep"], e["reason"][:18], e.get("R", 0.0), e.get("R_rate", 0.0),
                   ("%.3f" % e["hmean"]) if "hmean" in e else "-",
                   ("%.3f" % e["hmax"]) if "hmax" in e else "-",
                   ("%+.3f" % e["loss"]) if "loss" in e else "-",
                   ("%.3f" % e["entropy"]) if "entropy" in e else "-"))
        # --- xu the: 5 episode dau vs 5 episode cuoi cua phien ----------------------
        if len(eps) >= 4:
            head, tail = eps[:5], eps[-5:]

            def avg(rows, k):
                vals = [r[k] for r in rows if k in r]
                return sum(vals) / len(vals) if vals else float("nan")

            d = avg(tail, "R_rate") - avg(head, "R_rate")
            print("")
            print("    XU THE (5 episode DAU vs 5 episode CUOI cua phien):")
            print("      R/buoc    : %.3f -> %.3f  (%s%.3f)"
                  % (avg(head, "R_rate"), avg(tail, "R_rate"),
                     "+" if d >= 0 else "-", abs(d)))
            print("      do cao TB : %.3f -> %.3f" % (avg(head, "hmean"), avg(tail, "hmean")))
            print("      entropy   : %.3f -> %.3f  (giam = policy dang an toan hon)"
                  % (avg(head, "entropy"), avg(tail, "entropy")))
            n_fall = sum(1 for e in eps
                         if "NGA" in e["reason"].upper() or "FALL" in e["reason"].upper())
            print("      so episode bi NGA: %d/%d (%.0f%%)"
                  % (n_fall, len(eps), 100.0 * n_fall / len(eps)))
        if s.get("summary"):
            dur, nep, nstep, rate = s["summary"]
            print("    TONG KET phien: thoi gian %s | %s episode | %s buoc vat ly | %s buoc/giay"
                  % (dur, nep, nstep, rate))
        # --- bieu do text ------------------------------------------------------------
        if len(eps) >= 3:
            print("")
            print("    BIEU DO R_total theo episode:")
            print(ascii_chart([e.get("R", 0.0) for e in eps]))
    # --- cham diem chung ----------------------------------------------------------------
    rates = [e["R_rate"] for e in eps_all if "R_rate" in e]
    if rates:
        mean_rate = sum(rates) / len(rates)
        print("")
        print("  CHAM DIEM TONG (toan bo %d episode trong log):" % len(rates))
        print("    Reward TB / buoc : %.3f / toi da %.2f (%.0f%%) -> %s"
              % (mean_rate, MAX_STEP_REWARD, 100.0 * mean_rate / MAX_STEP_REWARD,
                 classify(mean_rate)))
    print(SEP)


# ====================================================================================
# PHAN 3 - PHAN TICH OFFLINE: POLICY "NGHI" GI KHI STATE DOI?
# ====================================================================================
def part3_offline(policy, stand, action_scale, n_noise=200):
    print("")
    print(SEP)
    print("PHAN 3: PHAN TICH OFFLINE - DO NHAY CUA POLICY (khong can env.exe)")
    print(SEP)
    home = make_home_state()

    # --- A: quet do cao hip_y (0.10 -> 0.26) ------------------------------------------
    print("  A) POLICY + REWARD THEO DO CAO hip_y (quet 0.10 -> 0.26):")
    print("     %-7s %-9s %-9s %-10s %-9s %s"
          % ("hip_y", "reward", "height_u", "|a| TB", "V(s)", "ghi chu"))
    hip_ys = [0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.19, 0.21, 0.23, 0.26]
    for hy in hip_ys:
        st = make_home_state(hy)
        a, _, value, reward, done, info = policy_eval(policy, st, stand, action_scale)
        note = "NGA (done)" if done else ""
        print("     %-7.3f %+9.3f %-9.2f %-10.4f %+9.3f %s"
              % (hy, reward, info["height_unit"], sum(abs(x) for x in a) / len(a),
                 value, note))

    # --- B: quet goc nghieng (body_sin -1 -> 1) ---------------------------------------
    print("")
    print("  B) POLICY + REWARD THEO GOC NGHIENG SONG LUNG (body_sin -1 -> 1):")
    print("     %-9s %-8s %-9s %-10s %s" % ("body_sin", "tilt_do", "upright", "reward", "a[hip]"))
    for i in range(9):
        bs = -1.0 + 0.25 * i
        st = make_home_state()
        st[eb.BONE_IDX["body"]] = bs
        a, _, _, reward, _, info = policy_eval(policy, st, stand, action_scale)
        print("     %-9.2f %-8.1f %-9.2f %+9.3f  %+.3f"
              % (bs, info["tilt_deg"], info["upright_unit"], reward, a[1]))

    # --- C: do do lon hanh dong tren nhieu state gan tu the dung ----------------------
    rng = random.Random(0)
    acts, vals, rewards = [], [], []
    for _ in range(n_noise):
        st = make_home_state()
        for i in range(len(st)):
            st[i] += rng.gauss(0.0, 0.05)
        st[eb.IDX_HIP_POS_Y] = max(0.05, min(0.30, st[eb.IDX_HIP_POS_Y]))
        a, _, value, reward, _, _ = policy_eval(policy, st, stand, action_scale)
        acts.append(a); vals.append(value); rewards.append(reward)
    n = float(len(acts))
    mean_abs = [sum(abs(x[j]) for x in acts) / n for j in range(len(acts[0]))]
    std_a = [math.sqrt(sum((x[j] - sum(y[j] for y in acts) / n) ** 2 for x in acts) / n)
             for j in range(len(acts[0]))]
    print("")
    print("  C) DO LON HANH DONG TREN %d STATE NHIEU (gauss 0.05 quanh tu the dung):" % n_noise)
    print("     %-10s %-9s %-9s" % ("khop", "|a| TB", "std(a)"))
    for j, name in enumerate(eb.JOINT_NAMES):
        print("     %-10s %9.4f %9.4f" % (name, mean_abs[j], std_a[j]))
    print("     V(s)          : TB=%+.3f  min=%+.3f  max=%+.3f"
          % (sum(vals) / n, min(vals), max(vals)))
    print("     reward state  : TB=%+.3f  min=%+.3f  max=%+.3f"
          % (sum(rewards) / n, min(rewards), max(rewards)))

    # --- nhan xet tu dong ---------------------------------------------------------------
    print("")
    print("  NHAN XET:")
    mean_overall = sum(mean_abs) / len(mean_abs)
    if mean_overall < 0.05:
        print("    - Policy giu gan nhu dung tu the STAND o moi state (|a| ~ %.3f):" % mean_overall)
        print("      bo xuong se dung yen, an diem thuan tu reward co so (height+upright+alive).")
    else:
        print("    - Policy co phan ung (|a| TB ~ %.3f): dieu chinh khop quanh tu the STAND."
              % mean_overall)
    hy_drop = make_home_state(0.13)
    a_low, _, _, _, _, _ = policy_eval(policy, hy_drop, stand, action_scale)
    hy_top = make_home_state(0.19)
    a_high, _, _, _, _, _ = policy_eval(policy, hy_top, stand, action_scale)
    diff = sum(abs(a_low[j] - a_high[j]) for j in range(len(a_low))) / len(a_low)
    print("    - Khac biet hanh dong khi hip_y thap (0.13) vs cao (0.19): %.4f" % diff)
    if diff > 0.02:
        print("      -> policy CO phan ung voi do cao (co gang sua tu the khi sup xuong).")
    else:
        print("      -> policy gan nhu KHONG phan ung voi do cao (co the dang 'tan tinh').")
    print(SEP)


# ====================================================================================
# PHAN 4 - LIVE EVAL: CHAY THAT TREN env.exe (deterministic, khong cap nhat trong so)
# ====================================================================================
def part4_live(policy, stand, action_scale, episodes=5, steps=300, csv_path=None):
    print("")
    print(SEP)
    print("PHAN 4: LIVE EVAL TREN env.exe (deterministic)")
    print(SEP)
    try:
        eb.open_sockets()
        eb.request_state()
    except (eb.EnvNotRunning, OSError) as exc:
        print("  [KHONG CHAY DUOC] %s" % exc)
        print("  Huong dan: mo env.exe truoc roi chay lai voi co --live.")
        print(SEP)
        return None

    writer = csv_file = None
    if csv_path:
        d = os.path.dirname(csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "step", "hip_y", "hip_x", "tilt_deg", "grounded",
                         "r_total", "r_h", "r_u", "r_s", "r_b", "r_c", "r_a",
                         "fall", "value", "done"])

    sum_rows = []
    try:
        for ep in range(episodes):
            eb.reset_env()
            state = list(eb.request_state())
            x_home = state[eb.IDX_HIP_POS_X]
            R = 0.0
            h_sum = h_max = 0.0
            a_abs_sum = 0.0
            end_reason = "HET BUOC"
            for step in range(1, steps + 1):
                eb.render_tick()
                a, _, value, _ = policy.act(
                    eb.normalize_state(state).unsqueeze(0), deterministic=True)
                target = eb.joint_target_from_action(a[0], stand, action_scale)
                eb.send(*target)
                state = list(eb.request_state())
                r, done, info = eb.compute_reward(state, x_home)
                R += r
                h_sum += info["hip_y"]
                h_max = max(h_max, info["hip_y"])
                a_abs_sum += sum(abs(float(x)) for x in a[0]) / len(a[0])
                if writer:
                    writer.writerow([ep, step, "%.4f" % info["hip_y"], "%.4f" % info["hip_x"],
                                     "%.2f" % info["tilt_deg"], info["grounded"],
                                     "%.4f" % r, "%.4f" % info["height"],
                                     "%.4f" % info["upright"], "%.4f" % info["stability"],
                                     "%.4f" % info["balance"], "%.4f" % info["center"],
                                     "%.4f" % info["alive"], "%.2f" % info["fall"],
                                     "%.4f" % float(value[0]), int(done)])
                if done:
                    end_reason = "NGA (hip_y=%.3f)" % info["hip_y"]
                    break
            rate = R / step
            sum_rows.append({"ep": ep, "R": R, "rate": rate, "steps": step,
                             "hmean": h_sum / step, "hmax": h_max,
                             "a_abs": a_abs_sum / step, "reason": end_reason})
            print("  EP %d/%d: %-18s R=%+8.2f | R/buoc=%.3f | buoc=%d/%d | "
                  "do cao TB=%.3f max=%.3f | |a|=%.3f"
                  % (ep + 1, episodes, end_reason, R, rate, step, steps,
                     h_sum / step, h_max, a_abs_sum / step))
            if writer:
                csv_file.flush()
    finally:
        if csv_file:
            csv_file.close()
            print("  Da xuat CSV tung buoc: %s" % csv_path)
        eb.close_sockets()
    return sum_rows


def live_summary(sum_rows, episodes, steps):
    n = len(sum_rows)
    if not n:
        return
    surv = sum(1 for r in sum_rows if r["steps"] >= steps)
    mean_rate = sum(r["rate"] for r in sum_rows) / n
    print("")
    print("  KET QUA LIVE EVAL (%d episode x %d buoc):" % (n, steps))
    print("    Dung tron vi tri : %d/%d (%.0f%%)" % (surv, n, 100.0 * surv / n))
    print("    Reward TB / buoc : %.3f / toi da %.2f (%.0f%%) -> %s"
          % (mean_rate, MAX_STEP_REWARD, 100.0 * mean_rate / MAX_STEP_REWARD,
             classify(mean_rate)))
    print("    Do cao hip TB    : %.4f | max %.4f (STAND ~ 0.169)"
          % (sum(r["hmean"] for r in sum_rows) / n,
             max(r["hmax"] for r in sum_rows)))
    print("    |a| TB           : %.4f (do lech hanh dong so voi tu the STAND)"
          % (sum(r["a_abs"] for r in sum_rows) / n))


# ====================================================================================
# MAIN
# ====================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Kiem tra mo hinh bo xuong da hoc (checkpoint + log + live eval).")
    p.add_argument("--checkpoint", default="rha_policy.pt",
                   help="duong dan checkpoint (mac dinh rha_policy.pt)")
    p.add_argument("--log", default="t1.log",
                   help="file log huan luyen de phan tich; '' de bo qua PHAN 2")
    p.add_argument("--detail", action="store_true",
                   help="in them thong ke trong so tung lop (PHAN 1)")
    p.add_argument("--live", action="store_true",
                   help="chay them PHAN 4: live eval tren env.exe (can mo env.exe truoc)")
    p.add_argument("--episodes", type=int, default=5,
                   help="so episode cho live eval (mac dinh 5)")
    p.add_argument("--steps", type=int, default=300,
                   help="so buoc toi da moi episode (mac dinh 300 ~ 5 giay)")
    p.add_argument("--csv", default="test_results/eval_steps.csv",
                   help="file CSV xuat tung buoc cua live eval ('' de tat)")
    p.add_argument("--noise-states", type=int, default=200,
                   help="so state nhieu cho PHAN 3C (mac dinh 200)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ckpt = resolve_path(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise SystemExit("[LOI] khong tim thay checkpoint: %s" % ckpt)

    policy, meta, stand, action_scale = part1_checkpoint(ckpt, detail=args.detail)

    if args.log:
        part2_log(resolve_path(args.log))

    part3_offline(policy, stand, action_scale, n_noise=args.noise_states)

    if args.live:
        rows = part4_live(policy, stand, action_scale, episodes=args.episodes,
                          steps=args.steps, csv_path=(args.csv or None))
        if rows is not None:
            live_summary(rows, args.episodes, args.steps)
    else:
        print("")
        print("  [GON Y] chay 'python src/env_test.py --live' (sau khi mo env.exe)")
        print("  de danh gia that su bo xuong trong mo phong.")


if __name__ == "__main__":
    main()






