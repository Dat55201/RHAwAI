# RHAwAI 🇻🇳

> **R**agdoll **H**umanoid **Aw**areness + **A**rtificial **I**ntelligence  
> Mô phỏng môi trường vật lý C++ (OpenGL) + huấn luyện Reinforcement Learning Python (PyTorch) giao tiếp qua UDP

---

## Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
3. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
4. [Cài đặt](#cài-đặt)
5. [Cách chạy nhanh](#cách-chạy-nhanh)
6. [Giao thức UDP](#giao-thức-udp)
7. [State & Action Space](#state--action-space)
8. [Hệ thống thưởng (Reward)](#hệ-thống-thưởng-reward)
9. [Mạng nơron & thuật toán RL](#mạng-nơron--thuật-toán-rl)
10. [Chi tiết hàm & thuật toán](#chi-tiết-hàm--thuật-toán)
11. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
12. [Điều khiển trong `env.exe`](#điều-khiển-trong-envexe)
13. [Checkpoint & Resume](#checkpoint--resume)
14. [Mẹo & Tips](#mẹo--tips)
15. [Credits & License](#credits--license)

---

## Giới thiệu

**RHAwAI** là một dự án *reinforcement learning* (RL) trên một **biped ragdoll** 11 xương, nơi:

- **Môi trường vật lý (C++)** — `env.exe` mô phỏng một bộ xương người hai chân (head, body, hip, 4 chi, 2 cánh tay, 2 gối) bằng OpenGL 2D, tính toán vật lý (gravity, rigid-body dynamics, joint constraints) và hiển thị real-time.
- **Agent (Python)** — `env_bridge.py` dùng một mạng MLP 2 lớp (actor-critic) với PyTorch, học điều khiển 10 khớp để thực hiện 3 kỹ năng:
  1. **Giữ thăng bằng** — `balance`
  2. **Đứng thẳng** — `upright`
  3. **Vượt cao nhất có thể** — `maximize height`

Hai thành phần giao tiếp bằng **giao thức UDP** (C++ là server, Python là client), giúp tách biệt hoàn toàn giữa mô phỏng vật lý (nhanh, phần cứng) và học máy (linh hoạt, dễ thử nghiệm).

---

## Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────┐
│                    RHAwAI — Data Flow                    │
├─────────────┐     UDP 5005 (command/action)    ┌─────────┤
│  Python     │ ───────────────────────────────→ │  C++    │
│  Agent      │                                    │  env.exe│
│  (PyTorch)  │ ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │  (OpenGL│
│             │       UDP 5006 (39-float state)   │  +Phys) │
└─────────────┘                               └───────────┘
```

**Vòng lặp huấn luyện (A2C):**

```
1.  policy.act(state)      ← nhận state 39 float từ env.exe
2.  action (10 targetAngle) → gửi qua UDP
3.  env.exe step(dt=1/60s) → tính physics, render
4.  reward = compute_reward(state)
5.  loss.backward() + Adam → cập nhật mạng (per-step)
6.  lặp lại cho đến hết thời gian / episode
```

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| **Hệ điều hành** | Windows 10/11 (dùng WinSock2) |
| **Bộ biên dịch C++** | MinGW-w64 / GCC ≥ 8 (C++17) |
| **Python** | ≥ 3.8 |
| **PyTorch** | `torch` (CPU edition đề xuất) |
| **Thư viện C++** | GLFW 3, GLAD (gl=3.3), GLM (header-only) |

> **Lưu ý:** Phiên bản MinGW đã dùng trong dự án:
> `BrechtSanders.WinLibs.POSIX.UCRT` (mingw64)

---

## Cài đặt

### 1. Biên dịch `env.exe` (C++)

**Cách A — VS Code (đề xuất):**
- Mở thư mục dự án trong VS Code.
- Nhấn `Ctrl + Shift + B` → chọn `C/C++: g++.exe build active file`.
- File `env.exe` sẽ được tạo ở thư mục gốc.

**Cách B — Dòng lệnh thủ công:**

```bash
g++ -g -std=c++17 -I./include -L./lib \
    src/env.cpp src/glad.c \
    -lglfw3dll -lws2_32 -lopengl32 -lgdi32 \
    -o env.exe
```

### 2. Cài PyTorch (Python)

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Cách chạy nhanh

> **Bước 0:** Mở `env.exe` **trước** (nó là server UDP). Trong cửa sổ env, nhấn **`P`** để bật UDP runtime.

| Mode | Mô tả | Lệnh ví dụ |
|---|---|---|
| `train` | Huấn luyện (A2C, cập nhật mỗi bước) | `python src/env_bridge.py --mode train --time 1h --no-render --save-every 25` |
| `train` (REINFORCE) | Huấn luyện (cập nhật 1 lần/episode) | `python src/env_bridge.py --mode train --update episode --episodes 300 --steps 300` |
| `eval` | Chạy thử policy đã học (không học) | `python src/env_bridge.py --mode eval --resume rha_policy.pt --time 1h --no-render` |
| `walk` | Đi bộ bằng DAO ĐỘNG sine | `python src/env_bridge.py --mode walk --freq 1.5 --amp 1.0 --frames 600` |
| `stand` | Đứng yên ở tư thế STAND | `python src/env_bridge.py --mode stand --frames 600` |

---

## Giao thức UDP

`env.exe` và `env_bridge.py` giao tiếp qua 2 cổng UDP:

| Hướng | Cổng | Nội dung |
|---|---|---|
| Python → `env.exe` | `127.0.0.1:5005` | Lệnh / Action |
| `env.exe` → Python | `127.0.0.1:5006` | State (39 float) |

### Lệnh từ Python → env.exe

Mỗi gói tin là một hoặc nhiều `float` (đóng gói bằng `struct.pack("<%df", ...)`):

| Float đầu tiên | Tham số thứ 2 | Ý nghĩa |
|---|---|---|
| `-100.0` | `0.0` | **Xin state** — env.exe trả lời bằng 39 float |
| `-150.0` | `0.0` | **Yêu cầu render** 1 frame (`glfwSwapBuffers`) |
| `-69.0` | `idx` | **Reset skeleton** |
| `-68.0` | `mag` | Đặt `impulse_max` (0 = tắt rối ngẫu nhiên) |
| `10 float` | — | **Action**: 10 `targetAngle` cho 10 khớp (theo thứ tự joints) |

### State từ env.exe → Python (39 float)

| Index | Nội dung |
|---|---|
| `0..32` (11×3) | 11 xương × `[sin(angle), cos(angle), angVel]` |
| `33` | `hip.pos.y / 600` (độ cao) |
| `34` | `hip.pos.x / 800` |
| `35` | `hip.vel.y / 600` |
| `36` | `hip.vel.x / 800` |
| `37` | `calfL` chạm đất (1.0/0.0) |
| `38` | `calfR` chạm đất (1.0/0.0) |

**Thứ tự 11 xương** trong state (trùng với `bones[]` trong env.cpp):
`head, body, armL, armR, forearmL, forearmR, legL, legR, calfL, calfR, hip`

---

## State & Action Space

### Observation (39 float)

| idx | bone | sin | cos | angVel | hip | grounded |
|-----|------|-----|-----|--------|-----|----------|
| 0–2 | head | 0 | 1 | 2 | | |
| 3–5 | body | 3 | 4 | 5 | | |
| 6–8 | armL | 6 | 7 | 8 | | |
| 9–11 | armR | 9 | 10 | 11 | | |
| 12–14 | forearmL | 12 | 13 | 14 | | |
| 15–17 | forearmR | 15 | 16 | 17 | | |
| 18–20 | legL | 18 | 19 | 20 | | |
| 21–23 | legR | 21 | 22 | 23 | | |
| 24–26 | calfL | 24 | 25 | 26 | | |
| 27–29 | calfR | 27 | 28 | 29 | | |
| 30–32 | hip | 30 | 31 | 32 | | |
| 33 | — | hip.pos.y / 600 | | | | |
| 34 | — | hip.pos.x / 800 | | | | |
| 35 | — | hip.vel.y / 600 | | | | |
| 36 | — | hip.vel.x / 800 | | | | |
| 37 | — | calfL chạm đất (0/1) | | | | |
| 38 | — | calfR chạm đất (0/1) | | | | |

> **Chuẩn hóa quan sát:** Mọi giá trị được ép về `[-1, 1]` trước khi đưa vào mạng:
> - `sin/cos` → giữ nguyên (đã trong `[-1, 1]`)
> - `angVel` → chia cho `ANG_VEL_SCALE = 20.0` (vì env.cpp giới hạn `[-50, 50]`)
> - `hip.pos` → nhân/scaling để về `[-1, 1]`
> - `hip.vel` → nhân/scaling
> - `ground contact` → giữ nguyên `0.0/1.0`

### Action (10 float)

```
THỨ TỰ 10 KHỚP (targetAngle, radian):  head, hip, armR, armL,
                                        forearmR, forearmL, legR, legL, calfR, calfL
```

Mỗi giá trị `targetAngle` là **góc tuyệt đối** giữa hai xương (thực tế dao động `0 → ~5 rad`).

**Ánh xạ từ action mạng sang targetAngle:**

$$targetAngle[i] = STAND[i] + action\_scale \times \tanh(policy[i])$$

Với `action_scale = 0.35` (mặc định) và `STAND`:

```python
STAND = [0.0, 4.95, 3.0, 4.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
#      head, hip, armR, armL, forearmR, forearmL, legR, legL, calfR, calfL
```


---

## Hệ thống thưởng (Reward)

`compute_reward()` trong `env_bridge.py` tính tổng thưởng từ **6 thành phần**:

$$R_{total} = w_h \cdot r_h + w_u \cdot r_u + w_s \cdot r_s + w_b \cdot r_b + w_c \cdot r_c + w_a - \mathbb{1}_{fall} \cdot P_{fall}$$

| Thành phần | Trọng số | Công thức | Ý nghĩa |
|---|---|---|---|
| **Height** | `W_HEIGHT = 1.0` | `clip((hip_y − HEIGHT_FLOOR) / (HEIGHT_REF − HEIGHT_FLOOR), 0, 1)` | Thưởng khi cao |
| **Upright** | `W_UPRIGHT = 0.6` | `max(0, 1 − tilt° / TILT_TOL)` | Thưởng khi thẳng |
| **Stability** | `W_STABILITY = 0.15` | `−(shaking + drift + bounce)` | Phạt rung lắc |
| **Balance** | `W_BALANCE = 0.3` | `1.0 (2 chân đất)`, `0.0 (1 chân)`, `−0.5 (0 chân)` | Thưởng thăng bằng |
| **Center** | `W_CENTER = 0.15` | `−min(1, \|hip_x − x_home\|)` | Phạt lệch trung tâm |
| **Alive** | `W_ALIVE = 0.10` | `0.10` | Thưởng sống mỗi bước |
| **Fall (penalty)** | `FALL_PENALTY = 10.0` | `−10` khi `hip_y < FALL_HEIGHT = 0.12` | Kết thúc episode |

**Tham số:**

```python
HEIGHT_FLOOR  = 0.13   # hip_y dưới này = 0 điểm chiều cao (sup xuống)
HEIGHT_REF    = 0.17   # hip_y ≥ ngưỡng này = điểm tối đa (vuột qua tầm vớ)
TILT_TOL_DEG  = 45.0   # góc nghiêng tối đa để được 1.0 điểm thẳng
ANG_VEL_SCALE = 20.0   # chuẩn hóa angVel về [-1, 1]
```

> 💡 **Chiến lược thiết kế:** Chiều cao được *bảo hòa* (clip) để tránh "lừa" bằng cách nhấc một chân — nếu không có bao hòa, policy sẽ ưu tiên +0.8 chiều cao nhưng mất −0.3 thăng bằng, dẫn đến ngã sớm. `body` (sống lồng) khởi tạo với góc 1.50 rad → `sin(body_angle) = 1.0` (thẳng đứng) — đây là cơ sở của Upright Reward.

---

## Mạng nơron & thuật toán RL

### Policy Network (MLP 2 lớp)

```
Input (39) → Linear(39, 128) → tanh → Linear(128, 128) → tanh → Heads:
         μ (tanh) → Linear(128, 10)    [actor: action means]
         V       → Linear(128, 1)      [critic: value baseline]
         log_std → Parameter(10)        [learnable, shared across steps]
```

- **Khởi tạo trọng số:** Orthogonal init (`gain=√2` cho hidden layers, `gain=0.01` cho μ-head → ban đầu μ ≈ 0 → action ≈ STAND)
- **log_std:** Khởi tạo `−1.4` rad (σ ≈ 0.25), giới hạn `[−3.5, −1.0]`
- **Forward:** `h = tanh(fc1(state))` → `h = tanh(fc2(h))` → `mu = tanh(mu_head(h))`, `value = value_head(h)`, `log_std` clamp

### Chính sách (Squashed Gaussian)

$$u \sim \mathcal{N}(\mu(s), \sigma),\quad a = \tanh(u),\quad \log\pi(a|s) = \log\mathcal{N}(u) - \log(1 - a^2 + \epsilon)$$

- `deterministic=True` (eval): `a = tanh(mu)` — greedy, không có nhiễu
- `deterministic=False` (train): mẫu `u` bằng `dist.rsample().detach()` (score-function estimator)

### Thuật toán

| Thuật toán | `--update` | Cập nhật | Loss |
|---|---|---|---|
| **A2C** (mặc định) | `online` | Mỗi bước | `L = −log_prob · Â + 0.5 · MSE(V, ĉ) / σ²_target − β · H(π)` |
| **REINFORCE** | `episode` | Mỗi episode | `L = −Σ log_prob · ĝ_t` |

**Công thức chi tiết:**

- **Discounted return:** $G_t = r_t + \gamma \cdot G_{t+1}$ với $\gamma = 0.995$
- **Advantage (A2C):** $\hat{A} = (r + \gamma \cdot V(s')) - V(s)$, chuẩn hóa qua `RunningStat` (Welford online)
- **Advantage (REINFORCE):** $\hat{A} = \hat{G} - V(s).\text{detach}()$, chuẩn hóa về zero-mean/unit-std
- **Critic target normalization:** Value loss chia cho phương sai target $\sigma^2_{target}$ để tránh gradient bị lệch
- **Tối ưu:** Adam ($\text{lr}=3\times10^{-4}$), gradient clipping $\|g\|_2 \leq 1.0$
- **Entropy coef:** $\beta = 0.002$ (nhỏ vì `log_std` dùng chung — tránh σ bùng lên mãi mãi)
- **Seeds cố định:** `--seed 1234` (có thể thay đổi để so sánh)

**Vòng lặp huấn luyện (A2C):**

```
for mỗi episode:
    reset_env() → x_home = hip_x ban đầu
    for t = 0..steps:
        1. state → policy.act() → (action, log_prob, value, entropy)
        2. send(targetAngle = STAND + scale × tanh(action)) → env.exe
        3. request_state() → raw_next
        4. reward = compute_reward(raw_next, x_home)
        5. [A2C] loss.backward() + Adam.step()  |  [REINFORCE] lưu buf, cập nhật cuối episode
        6. log tiến độ (mỗi log_every bước)
        7. raw = raw_next; time.sleep(FRAME_DT=1/60)
        [REINFORCE] tính G_t, advantage, loss.backward() 1 lần
```

---

## Chi tiết hàm & thuật toán

### A. `env.cpp` — Các struct & hàm chính

#### `Engine` — Cửa sổ & renderer

| Hàm | Mô tả |
|---|---|
| `Engine()` | `glfwInit()` → cửa sổ 800×600 ("RL is pain") → `glfwMakeContextCurrent` → `glViewport` |
| `run()` | `glClear(GL_COLOR_BUFFER_BIT)` + `glOrtho(0, 800, 0, 600, -1, 1)` (2D orthographic) |
| `cross(vec2, vec2)` | Tích vectơ 2D: $a_x b_y - a_y b_x$ |
| `drawCircle(pos, r, color)` | Vẽ hình tròn bằng `GL_TRIANGLE_FAN` (20 mạch) |
| `rotate(v, a)` | Quay 2D: $(c x - s y,\; s x + c y)$ |

#### `Bone` — Thân ragdoll (khối lượng rắn)

| Thuộc tính | Giá trị / Công thức |
|---|---|
| `pos, vel` | Vị trí & vận tốc tâm (vec2) |
| `angle, angVel` | Góc (rad) & tốc độ góc |
| `halfLength, radius` | Kích thước hình chữ nhật quay |
| `mass, inertia` | $I = \frac{m(L^2 + W^2)}{12}$ |
| `invMass, invInertia` | $\frac{1}{m},\; \frac{1}{I}$ |
| `force, torque` | Lực & mô-men tích tụ |
| `dragged, highlight` | Trạng thái kéo / highlight màu cam |

| Hàm | Mô tả |
|---|---|
| `worldPoint(local)` | Biến đổi local → world qua ma trận quay 2D |
| `draw()` | Render `GL_QUADS` thân + 2 `GL_TRIANGLE_FAN` đầu (highlight = cam) |

#### `Joint` — Khớp nối (revolute) với motor PD

| Thuộc tính | Giá trị |
|---|---|
| `A, B` | Hai Bone nối với nhau |
| `anchorA_local, anchorB_local` | Điểm neo trong hệ local của A, B |
| `maxTorque` | $1 \times 10^5$ (giới hạn) |
| `targetAngle` | Góc mục tiêu (do Python gửi qua UDP) |
| `stiffness` | Hệ số cứng (mặc định 1.0) |

| Hàm | Mô tả | Thuật toán |
|---|---|---|
| `solve(dt)` | Ràng buộc vị trí | **Baumgarte stabilization** (β = 0.8): <br> $C = p_B - p_A$, $bias = \frac{\beta}{\Delta t} C$, $Cdot = v_B - v_A + bias$<br>Giải hệ 2×2 → impulse → áp dụng lên A, B |
| `applyTorque(dt)` | Điều khiển góc về targetAngle | **PD controller**: <br> $error = targetAngle - (B.\theta - A.\theta)$ (wrapped $[-π, π]$)<br>$k = stiffness \cdot maxTorque$, $d = 2\sqrt{k(I_A + I_B)}$<br>$torque = k \cdot error - d \cdot \dot{error}$, clamp ±maxTorque |

#### `Skeleton` — Bộ xương 11 (10 khớp)

| Trường | Giá trị / Mô tả |
|---|---|
| `bones` | Vector 11 Bone: head, body, hip, armL, armR, forearmL, forearmR, legL, legR, calfL, calfR |
| `joints` | Vector 10 Joint (revolute) |
| `startPos` | Vị trí ban đầu (400, 60) |
| `impulse_max` | Độ lớn rối ngẫu nhiên (0 = tắt, dùng cho training) |

| Hàm | Mô tả |
|---|---|
| `init(p, jitter)` | Tạo 11 Bone với góc khởi tạo cụ thể (body=1.50 rad, hip=4.95 rad, …) + jitter ngẫu nhiên |
| `step(dt)` | Euler: $v \mathrel{+}= g\cdot dt$; Damping: $\times\frac{1}{1+0.7\cdot dt}$; 50× solve; torque; ground collision; foot friction ($v_x \times 0.05$); ApplyImpulse |
| `ApplyImpulse()` | Rối ngẫu nhiên (exploration): chọn Bone theo trọng số `[body:0.5, head:0.2, arm:0.1, leg:0.05]`, delay 60–180 frame |
| `reset()` | Gọi `init(startPos, 0.3f)` |
| `checkBorderCollision(b)` | Sàn: restitution=0.2, slop=0.01, percent=0.8, friction=0.99 |


#### `DataManager` — Giao tiếp UDP

| Hàm | Mô tả |
|---|---|
| `init()` | `WSAStartup` → tạo socket UDP → `bind(5005)` → lưu `sockaddr_in` của Python |
| `receiveData(w)` | `recvfrom()` trên port 5005: <br>• `-100.0` → trả state qua `sendData()`<br>• `-150.0` → `glfwSwapBuffers()`<br>• `-69.0 + idx` → `sk->reset()`<br>• `-68.0 + mag` → đặt `impulse_max`<br>• 10 float → gán `targetAngle` cho 10 joints<br>Kiểm tra phím `P` để bật/tắt runtime |
| `sendData()` | Gửi 39 float state → `127.0.0.1:5006` |


#### `main()` — Vòng lặp chính

```cpp
srand(seed);
sk->impulse_max = 0;              // tắt rối ngẫu nhiên cho training
glfwSwapInterval(1);
while (!glfwWindowShouldClose(window)) {
    engine.run();                     // clear + ortho
    tempKeyControl(window);           // input bàn phím
    mouseCtl.update(window, dt);      // input chuột
    gotState = dataManager.receiveData(window);  // UDP nhận lệnh
    sk->step(dt);                     // physics step (dt=1/60)
    circleMass.update(window, dt);    // interactive mass
    if (gotState) dataManager.sendData();  // UDP gửi state
    if (timer % frameInterval == 0)
        glfwSwapBuffers(window);      // render (theo frameInterval)
    glfwPollEvents();
}
```

### B. `env_bridge.py` — Các hàm chính

#### Giao tiếp UDP

| Hàm | Mô tả |
|---|---|
| `open_sockets()` | Tạo/bind 2 socket: `state_sock` (bind 5006), `cmd_sock` (gửi 5005) |
| `send(*vals)` | `struct.pack("<%df", vals)` → gửi qua `cmd_sock` |
| `_drain_state_socket()` | Xóa state cũ trong buffer (tránh lag → reward sai lệch) |
| `request_state()` | Gửi `CMD_STATE` → `recvfrom` 39 float → unpack tuple |
| `request_state_retry(seconds)` | Tự động thử lại mỗi 2s, tối đa `retry_seconds` (dành cho pháo 10h) |
| `reset_env()` | `send(CMD_RESET, 0.0)` — reset về tư thế ban đầu |
| `set_impulse(mag)` | `send(CMD_IMPULSE, mag)` — bật/tắt rối ngẫu nhiên |
| `render_tick(enabled)` | `send(CMD_RENDER, 0.0)` — yêu cầu env.exe render 1 frame |

#### Học máy & mạng

| Hàm / Lớp | Mô tả |
|---|---|
| `PolicyNet` | MLP 2 lớp: `Linear(39,h)+tanh → Linear(h,h)+tanh → [μ:10, V:1]` |
| `PolicyNet.forward(s)` | Trả về `(mu, log_std, value)` |
| `PolicyNet.act(s, deterministic)` | Squashed Gaussian: `a = tanh(u)`, tính `log_prob` + `entropy` |
| `joint_target_from_action(a, stand, scale)` | `target[i] = STAND[i] + scale × a[i]` |
| `normalize_state(raw, device)` | Ép observation về `[-1, 1]` |
| `compute_reward(raw, x_home)` | Tính reward 6 thành phần + fall detection |
| `save_checkpoint(path, policy, meta)` | `torch.save({state_dict, meta})` |
| `load_checkpoint(path, device)` | Trả về `(meta, state_dict)` |
| `build_policy(cfg, device)` | Tạo PolicyNet, nạp checkpoint nếu có `--resume` |

#### Vòng lặp & chế độ chạy

| Hàm | Mô tả |
|---|---|
| `run_training(cfg)` | A2C/episode: interact → loss.backward() → Adam.step() |
| `run_eval(cfg)` | Deterministic (greedy) — không cập nhật |
| `run_walk(cfg)` | Sin gait: `STAND[i] + AMP[i]·amp·sin(2πft + φ[i])` |
| `run_stand(cfg)` | Gửi STAND liên tục |
| `run_timed(cfg, seconds)` | Chạy trong `seconds` giây thực, tự dừng + lưu checkpoint |
| `parse_args(argv)` | argparse: `--mode`, `--time`, `--episodes`, `--lr`, `—hidden`, … |
| `main(argv)` | Entry point: mở socket → dispatch mode → đóng socket |

#### Tiện ích

| Hàm / Lớp | Mô tả |
|---|---|
| `parse_duration(text)` | `'1h'→3600`, `'30m'→1800`, `'90s'→90`, `'45'→2700 (phút)` |
| `hhmmss(seconds)` | Định dạng `HH:MM:SS` |
| `TimeBudget` | Quản lý thời gian thực: `elapsed`, `remaining`, `expired()`, `progress_line()` |
| `RunningStat` | Welford online mean/std — chuẩn hóa advantage & critic target |
| `Window(size)` | Trung bình cộng N giá trị gần nhất |
| `log(msg)` | `print(msg, flush=True)` — realtime |

### C. Thuật toán vật lý chi tiết (`Skeleton::step`)

```
BƯỚC 1 — Euler Integration:
    v += g * dt                     (g = (0, -980.6) cm/s²)
    v *= 1 / (1 + 0.7 * dt)         (linear damping)
    ω *= 1 / (1 + 0.7 * dt)         (angular damping)

BƯỚC 2 — Constraint Resolution (x50 iterations):
    for mỗi Joint: solve(dt)         → Baumgarte stabilization (β=0.8)
    for mỗi Joint: applyTorque(dt)   → PD motor (k=stiffness·maxT, d=2√(k·(I_A+I_B)))

BƯỚC 3 — Ground Collision:
    for mỗi Bone: checkBorderCollision(b)
        → vận tốc tiếp tuyến, restitution=0.2, positional correction

BƯỚC 4 — Foot Friction:
    calfL, calfR: nếu chạm đất → v_x *= 0.05

BƯỚC 5 — Random Impulse:
    ApplyImpulse() — khám phá (chỉ khi impulse_max > 0)
```

**Baumgarte Stabilization** (trong `Joint::solve`):
$$C = p_B - p_A,\quad bias = \frac{\beta}{\Delta t} C,\quad \dot{C} = v_B - v_A + bias$$
$$\text{impulse} = -K^{-1} \dot{C},\quad K = \begin{bmatrix} k_{11} & k_{12} \\ k_{21} & k_{22} \end{bmatrix}$$

**PD Motor** (trong `Joint::applyTorque`):
$$error = targetAngle - (B.\theta - A.\theta) \quad (\text{wrapped } [-\pi, \pi])$$
$$\tau = k \cdot error - d \cdot \dot{error},\quad k = stiffness \cdot maxTorque,\quad d = 2\sqrt{k(I_A + I_B)}$$
$$\tau_{clamp} \in [-maxTorque, +maxTorque],\quad impulse = \tau \cdot dt$$

### D. Thuật toán RL chi tiết

**A2C (online, `--update online`):**

$$target = r + \gamma \cdot V(s'),\quad \hat{A} = (target - V(s)),\quad \hat{A}_{norm} = \frac{\hat{A} - \mu_A}{\sigma_A}$$
$$\mathcal{L} = -\log\pi(a|s)\cdot\hat{A}_{norm} + 0.5\cdot\frac{MSE(V, target)}{\sigma^2_{target}} - \beta\cdot H(\pi)$$

**REINFORCE (episode, `--update episode`):**

$$G_t = r_t + \gamma \cdot G_{t+1},\quad \hat{G} = \frac{G - \mu_G}{\sigma_G}$$
$$\hat{A} = \hat{G} - V(s).\text{detach}(),\quad \mathcal{L} = -\log\pi(a|s)\cdot\hat{A}$$

| Tham số | Giá trị mặc định | Mô tả |
|---|---|---|
| `lr` | `3e-4` | Learning rate của Adam |
| `gamma` | `0.995` | Hệ số chiết khấu (~nhìn 200 bước = 3.3s) |
| `hidden` | `128` | Số neuron mỗi lớp ẩn |
| `action_scale` | `0.35` | Biên độ lệch so với STAND (rad) |
| `entropy_coef` (β) | `0.002` | Hệ số khám phá (nhỏ vì log_std dùng chung) |
| `max_grad_norm` | `1.0` | Gradient clipping L2-norm |
| `seed` | `1234` | Seed PyTorch |
| `log_std_init` | `-1.4` | σ ≈ 0.25 rad |
| `log_std_min/max` | `-3.5 / -1.0` | Giới hạn σ |

---

## Cấu trúc thư mục

```
RHAwAI/
├── README.md                  ← Tài liệu dự án (file này)
├── env.exe                    ← Binary C++ (đã biên dịch sẵn)
├── glfw3.dll                  ← GLFW runtime (cần có khi chạy)
├── libgcc_s_seh-1.dll         ← MinGW runtime
├── libstdc++-6.dll            ← MinGW runtime
├── libwinpthread-1.dll        ← MinGW runtime
├── lib/
│   └── libglfw3dll.a          ← GLFW import library
├── include/
│   ├── GLFW/                  ← GLFW headers (glfw3.h, glfw3native.h)
│   ├── glad/                  ← GLAD OpenGL loader (glad.h)
│   ├── KHR/                   ← KHR platform (khrplatform.h)
│   └── glm/                   ← GLM math (header-only, ~1000 file)
├── src/
│   ├── env.cpp                ← Engine C++ (physics + render + UDP)
│   ├── env_bridge.py          ← Agent Python (PyTorch RL)
│   └── glad.c                 ← GLAD loader source
└── .vscode/
    ├── c_cpp_properties.json  ← Cấu hình IntelliSense (g++ C++17)
    ├── settings.json          ← Cấu hình CMake source dir
    └── tasks.json             ← Task build (Ctrl+Shift+B)
```

---

## Điều khiển trong `env.exe`

### Bàn phím

| Phím | Chức năng |
|---|---|
| `P` | **Bật/tắt runtime UDP** — bắt buộc bật để Python kết nối |
| `Y` | Đổi `frameInterval` (1 = 60 FPS ↔ 8 = 7.5 FPS) — tăng tốc khi không render |
| `R` | Reset skeleton về tư thế ban đầu |
| `↑/↓` (mũi tên) | Chọn khớp (joint index 0–9) |
| `←/→` (mũi tên) | Điều chỉnh `targetAngle` của khớp chọn (±0.005 rad) |
| `1–9, 0` | Áp dụng impulse (1500–10000) lên khớp chọn, hướng về chuột |
| `C` | Tạo khối tương tác (CircleMass) tại chuột |
| `V` | Xóa khối tương tác |

### Chuột

| Nút | Chế độ | Hành động |
|---|---|---|
| Trái (trên khớp) | **Rotate Joint** | Kéo để xoay khớp, `targetAngle` cập nhật theo góc chuột |
| Trái (trên xương) | **Drag Bone** | Kéo để kéo xương bằng velocity target (max 4000/s) |
| Phải (trên xương) | **Hard Rotate** | Xoay trực tiếp góc xương |
| Trái (trên CircleMass) | **Drag Mass** | Kéo khối tương tác |

---

## Checkpoint & Resume

### Định dạng `rha_policy.pt`

```python
{
    "state_dict": <trọng số mạng>,
    "meta": {
        "stand": [...],              # STAND vector (có thể thay đổi)
        "action_scale": 0.35,
        "hidden": 128,
        "state_dim": 39,
        "action_dim": 10,
        "saved_episode": 250,
        "best_mean_height": 0.169,
    }
}
```

### Cách dùng

| Lệnh | Mô tả |
|---|---|
| `--resume rha_policy.pt` | Nạp checkpoint (tiếp tục từ `saved_episode + 1`) |
| `--save rha_policy.pt` | Đường dẫn lưu (mặc định) |
| `--save ''` | Không lưu checkpoint |
| `--save-every 20` | Lưu checkpoint mỗi N episode |
| **Tự động backup** | Mỗi lần lưu, bản cũ được đổi thành `rha_policy.pt.bak` trước khi ghi mới |

> **Lưu ý:** Checkpoint luôn được lưu trong khối `finally` — kể cả khi người dùng nhấn Ctrl+C hoặc mất kết nối env.exe.

---

## Mẹo & Tips

### Tăng tốc huấn luyện

| Tùy chọn | Tác dụng |
|---|---|
| `--no-render` | Không gửi `CMD_RENDER` → nhanh hơn (tránh đồng bộ GPU) |
| `Y` trong env.exe | Đặt `frameInterval = 8` → chạy 8× nhanh hơn |
| `--quiet` | Bỏ log từng bước, chỉ giữ tổng kết episode |
| `--log-every 100` | Giảm tần suất log để tiết kiệm I/O |

### Chạy lâu dài (10h không ngừng)

```bash
python src/env_bridge.py --mode train --time 10h --no-render --save-every 20 --log-every 200
```

- `--retry 120`: Tự động thử lại nếu env.exe ngắt kết nối (tối đa 120s)
- Ctrl+C: Lưu checkpoint trước khi thoát
- `.bak` backup: Tránh mất checkpoint khi ghi đồng thời
- `--resume rha_policy.pt`: Tiếp tục sau khi khởi động lại

### Chạy theo giờ

```bash
python src/env_bridge.py --mode train --time 1h --no-render --save-every 25 --log-every 100
```

Hệ thống sẽ **tự động dừng** và **lưu checkpoint** khi hết giờ (kể cả giữa episode).

---

### Thư viện sử dụng

| Thư viện | Vai trò | Nguồn |
|---|---|---|
| **GLFW 3** | Tạo cửa sổ + input | <https://www.glfw.org/> |
| **GLAD** | OpenGL loader (gl=3.3 core) | <https://glad.dav1d.de/> |
| **GLM** | Toán thức (vector, góc, ma trận) | <https://github.com/g-truc/glm> |
| **PyTorch** | Mạng nơron + RL | <https://pytorch.org/> |
| **MinGW-w64** | Bộ biên dịch Windows | <https://www.mingw-w64.org/> |
---
