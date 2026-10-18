# CHANGELOG — EvoComm-MOMGRAP

Nhật ký thay đổi cho phần mở rộng Multi-Agent Learning + Evolutionary Communication
xây trên codebase `AAMAS_new / momgrap`. Mọi thay đổi phải giữ `python smoke_test.py`
tiếp tục qua và không phá vỡ API `momgrap` hiện có.

Định dạng: mỗi task một mục, ghi rõ file tạo/sửa, thay đổi API (nếu có), test thêm.

---

## Task 00 — Codebase snapshot & legacy regression guard

**Trạng thái:** hoàn thành. Thuần additive — không sửa file `momgrap/*.py` nào.

### File tạo
- `docs/CODEBASE_AUDIT_EVOCOMM.md` — snapshot commit/env, module map, `Config`
  defaults, và chữ ký 10 hàm public lõi (`sample_environment`, `build_comm_channels`,
  `bd_precoding`, `power_allocation`, `evm_sinr_rate`, `effective_steering`,
  `beampattern`, `decode`, `evaluate_single_regime`, `run_mfea`).
- `tests_marl/test_legacy_smoke.py` — regression guard nhẹ:
  - `test_single_regime_objective_sanity`: `evaluate_single_regime` trên LoS snapshot
    → assert shape `(N,2)`, finite, worst-user rate > 0, sensing không phẳng.
  - `test_tiny_mfea_archive_nonempty`: `run_mfea` với config cực nhỏ
    (`pop_size=12, max_gen=2, mc_samples=2, log_every=0`) → archive không rỗng,
    per-regime fronts đầy đủ.
- `scripts_marl/smoke_backend.py` — chạy chuỗi forward vật lý (decode →
  build_comm_channels → bd_precoding → power_allocation → evm_sinr_rate →
  effective_steering → beampattern) cho cả hai `power_policy` và in tensor shapes.
- `CHANGELOG_EVOCOMM.md` — file này.

### Thay đổi API
Không có. Task 00 không sửa `momgrap/`.

### Lệnh nghiệm thu
- `python smoke_test.py`
- `python scripts_marl/smoke_backend.py`
- `pytest -q tests_marl/test_legacy_smoke.py`

---

## Task 01 — Dependencies & MARL config riêng

**Trạng thái:** hoàn thành. Additive — không sửa `momgrap.config.Config` hay bất kỳ
file `momgrap/*.py` lõi nào; chỉ thêm subpackage mới `momgrap/marl/`.

### File tạo
- `requirements-marl.txt` — kế thừa `requirements.txt` + gymnasium/pettingzoo/
  PyYAML/tensorboard/pytest/scipy (không RLlib/Ray trong MVP).
- `pytest.ini` — `testpaths=tests_marl`, `pythonpath=.`, đăng ký marker `slow`/`cuda`
  (CI dùng `-m "not slow and not cuda"`).
- `momgrap/marl/__init__.py` — export config API (lazy với các dep nặng ở task sau).
- `momgrap/marl/config.py` — 7 frozen dataclass (`DynamicsConfig`, `CSIConfig`,
  `ActionConfig`, `CommunicationConfig`, `RewardConfig`, `RMAPPOConfig`,
  `EvoCommConfig`) + `load_evocomm_config(path, overrides)` +
  `evocomm_config_to_dict` / `save_resolved_config` + `resolve_dtype`.
  - physical block → `default_config(**physical)` (không mutate global config);
    `dtype_real` nhận string; grid mismatch → `ValueError` (bắt AssertionError).
  - Validation bằng `ValueError` tường minh: tau∈[0,1] & min≤max, delay≥0,
    rho∈[0,1), episode_length>0, ma_update_period>0, ris_group_count>0 &
    `L % ris_group_count == 0`, message budget≥0, regime_sampling/noise_scaling/
    graph_type hợp lệ, sanity RMAPPO.
- `configs_marl/smoke.yaml`, `configs_marl/mvp.yaml` — đúng Sec 25 / Sec 26.
- `tests_marl/test_config.py` — 27 test.

### Thay đổi API
Thêm mới (không phá vỡ cái cũ): `momgrap.marl.config.load_evocomm_config` và các
dataclass config. `momgrap.config.Config` giữ nguyên.

### Tests thêm (27)
Load smoke/mvp đúng giá trị; grid consistency; deep-merge overrides không mutate
file; `dtype_real` string; frozen (immutable); global config không đổi;
14 case validation reject (`ValueError`); missing section / unknown key / missing
file; serializable round-trip + `save_resolved_config`.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_config.py` → **27 passed**.
- `python -m pytest -q -m "not slow and not cuda"` (toàn bộ tests_marl) → **29 passed**.
- Import sanity: `momgrap` + `momgrap.marl` OK, `default_config().M == 16` (không mutate).

### Ghi chú
- `gymnasium` / `pettingzoo` / `tensorboard` chưa cài (Task 01 không cần). Sẽ cài
  qua `pip install -r requirements-marl.txt` khi tới Task 10 (pettingzoo) / Task 15
  (tensorboard).
- Không chạy lại `smoke_test.py` đầy đủ vì Task 01 không chạm file lõi; đã dựa vào
  `test_legacy_smoke.py` (chạy trong suite, qua) để bảo đảm không hồi quy và tránh
  ghi đè `figures/`.

---

## Task 02 — Backward-compatible channel component API

**Trạng thái:** hoàn thành. Chạm file lõi `momgrap/channels.py` (thay đổi tối thiểu,
backward-compatible bit-for-bit).

### File sửa
- `momgrap/channels.py` (+44/-5, chỉ vùng `build_comm_channels`):
  - Thêm dataclass `CommChannelComponents` (`direct`, `bs_ris`, `ris_user`,
    `reflected`, `aggregate`).
  - Thêm hàm public `build_comm_channel_components(P_ant, theta, env, cfg)` — dùng
    lại nguyên vẹn `_direct_channel` / `_bs_ris_channel` / `_ris_user_channel` và
    đúng chuỗi phép toán cũ.
  - `build_comm_channels()` trở thành wrapper: `return build_comm_channel_components(...).aggregate`.
  - Không đổi steering / path loss / LoS-NLoS mixing / dtype / device; không thêm randomness.

### File tạo
- `momgrap/marl/physics/__init__.py`, `momgrap/marl/physics/components.py` —
  re-export `CommChannelComponents` + `build_comm_channel_components` dưới namespace
  `momgrap.marl.physics` (không re-implement).
- `tests_marl/test_physics_components.py` — 7 test.

### Thay đổi API
- Thêm `momgrap.channels.CommChannelComponents` và `build_comm_channel_components`
  (cũng truy cập được qua `momgrap.marl.physics`).
- `build_comm_channels` giữ nguyên chữ ký và **output khớp bit-for-bit** (đã xác minh
  vs reference chụp trước khi sửa, `torch.equal == True` trên cả 3 regime).

### Shapes (xác minh, smoke geometry M=4/L=16/K=2, N=6, S=3)
`direct (6,3,2,4)`, `bs_ris (6,3,16,4)`, `ris_user (3,2,16)`, `reflected (6,3,2,4)`,
`aggregate (6,3,2,4)`; `aggregate == direct + reflected` (torch.equal).

### Tests thêm (7)
aggregate == legacy wrapper (bit-exact, 3 regime); aggregate == direct+reflected;
shapes N>1,S>1; finite & complex; CPU; determinism (không randomness);
marl re-export là đúng object lõi (identity).

### Lệnh đã chạy — kết quả
- Bit-for-bit vs reference-before-change: **ALL OK** (3/3 regime `torch.equal`).
- `python -m pytest -q tests_marl/test_physics_components.py` → **7 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **36 passed**.
- `python smoke_test.py` → **ALL SMOKE STAGES PASSED** (đã backup/restore `figures/`
  + `results/`; md5 khớp 100% sau khôi phục).

---

## Task 03 — True channel & estimated CSI

**Trạng thái:** hoàn thành. Additive — chỉ thêm file trong `momgrap/marl/physics/`,
không sửa file lõi.

### File tạo
- `momgrap/marl/physics/csi.py`:
  - `EstimatedChannels` (direct/reflected/aggregate/tau) — **không** chứa true channel.
  - `apply_complex_csi_error(true_tensor, tau, generator, scaling="rms")` —
    `Ĥ = √(1-τ²)·H + τ·Z_H`, `Z_H` circular complex Gaussian (E|Z|²=1) nhân
    `√(mean(|H|²))` (scaling "rms") hoặc đơn vị ("unit"). `τ=0` → clone đúng truth;
    giữ nguyên dtype/device; validate `τ∈[0,1]`.
  - `estimate_channel_components(true_components, tau, generator, scaling="rms")` —
    ước lượng direct & reflected độc lập, `aggregate = direct̂ + reflected̂` (nhất quán
    noise realization giữa BD-input và observation).
  - `CSIHistoryBuffer` (`reset`/`append`/`get_delayed`) — khi thiếu lịch sử thì **lặp
    estimate đầu tiên**, không zero-pad; `maxlen` tùy chọn.
- `tests_marl/test_csi.py` — 12 test.

### File sửa
- `momgrap/marl/physics/__init__.py` — export symbol CSI.
- `CHANGELOG_EVOCOMM.md`.

### Mô hình lỗi (dạng đóng, dùng để test)
Với RMS scaling và `Z_H ⟂ H`: `NMSE(τ) = 2(1 - √(1-τ²))` (0 tại τ=0, 2 tại τ=1),
tăng đơn điệu theo τ. Test khớp công thức trong sai số 15%.

### Quy tắc không leak (guard bằng test)
`W = bd_precoding(H_est, cfg)` (BD dùng **estimate**); `R = evm_sinr_rate(H_true, W, cfg)`
(realized rate dùng **truth**). Integration test chứng minh `W_est ≠ W_true` và tại
τ=0 thì `W_est == W_true`.

### Tests thêm (12)
τ=0 = truth; NMSE khớp `2(1-√(1-τ²))`; rel-error tăng theo τ (direct/reflected/
aggregate); deterministic cùng seed & khác khi khác seed; giữ complex64/complex128;
reject τ ngoài [0,1] & scaling lạ & non-complex; delay buffer (lặp first, clamp,
maxlen, lỗi empty/negative); `EstimatedChannels` không có field true; getter chỉ trả
estimate nhiễu; integration BD-estimate / rate-truth.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_csi.py` → **12 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **48 passed**.
- Không chạm file lõi → dựa vào legacy regression test trong suite (qua); không chạy
  lại `smoke_test.py` (tránh ghi đè `figures/`).

---

## Task 04 — Temporal dynamics

**Trạng thái:** hoàn thành. Additive — chỉ thêm file trong `momgrap/marl/`, không sửa lõi.

### File tạo
- `momgrap/marl/types.py` — `DynamicPhysicalState` (step_index, regime_index,
  environment, user_velocity, positions (1,M,3), theta (1,L), power_alloc (1,1,K),
  previous_physical_actions).
- `momgrap/marl/physics/dynamics.py` — `DynamicChannelProcess.reset(seed, positions,
  theta[, power])` / `.advance(state)`:
  - regime theo `regime_sampling` (fixed_los/rayleigh/rician theo tên regime, hoặc
    uniform_episode qua RNG episode); `sample_environment(regime, S=1, gen, cfg)`.
  - **User mobility** hằng vận tốc `p_{t+1}=p_t+v·dt` (mặt phẳng z=0), phản xạ vận tốc
    tại biên bán kính `[r_min,r_max]` & sector góc + clamp cứng ⇒ luôn trong vùng;
    sau khi dời user tính lại `user_dir/user_amp/g_los_dir/g_amp` (tái dùng
    `_pathloss_amp`, đúng casting float64 như `sample_environment`).
  - **Fading Gauss–Markov** `X_{t+1}=ρX_t+√(1-ρ²)ε` (ε ~ `_cn` CN(0,1)) trên
    `scat_gain_bu`, `G_nlos`, `g_nlos`; giữ marginal CN(0,1).
  - Cố định trong episode: regime, scatter directions, target geometry.
  - RNG episode riêng (seed ở reset), không dùng global RNG; `advance` không mutate
    state cũ (tạo state mới, copy dict actions).
- `tests_marl/test_dynamics.py` — 9 test.

### File sửa
- `momgrap/marl/physics/__init__.py` — export `DynamicPhysicalState`,
  `DynamicChannelProcess`.
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (9)
same seed → same trajectory; khác seed → khác; corr(ρ=0.99) > 0.9 và > corr(ρ=0)
(|corr(0)|<0.3); user luôn trong region qua 1000 bước (bán kính + sector + z=0);
geometry fields nhất quán (allclose với công thức tái dựng); 1000 transitions finite;
advance không mutate state cũ (+ dict copy, scatter fixed, fading đổi); mobility off
giữ user cố định; chọn regime cố định đúng tên.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_dynamics.py` → **9 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **57 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua); không chạy
  `smoke_test.py` (tránh ghi đè `figures/`).

---

## Task 05 — Physical action codebooks

**Trạng thái:** hoàn thành. Additive — chỉ thêm `momgrap/marl/actions/`, không sửa lõi.

### File tạo
- `momgrap/marl/actions/codebooks.py` — `AGENTS`, `MA_PATTERNS` (9), count helpers
  (`power_action_count`, `ris_action_count = 2G+3`, `ma_action_count = 9`),
  `power_profile_specs(K, size)`, `build_ris_group_map(cfg, action_cfg)`.
- `momgrap/marl/actions/decoder.py`:
  - `decode_power_action(idx, H_est, W_est, cfg, action_cfg, min_power_fraction=0.05)`
    → `p_user (1,1,K)`: profiles equal / inv-gain(α=0.5,1,2) / blend / focus-user
    (theo K), dựa trên **estimated** effective gains `|h_k^H w_k|²`; water-fill giữ
    floor và tổng = `p_max_watt`; không đụng `Config.power_policy`.
  - `decode_ris_action(current_theta, idx, group_map, phase_step)` → `(theta_next,
    n_changed)`: 0 no-op (exact), 1..2G ±Δ theo nhóm, 2G+1/2G+2 global ±Δ; wrap
    `[0,2π)`; chỉ nhóm được chọn thay đổi.
  - `decode_ma_action(current_positions, idx, cfg, action_cfg)` → `P_candidate (1,M,3)`
    = current + offset pattern (expand/contract y,z; shear ±; checkerboard y,z);
    x giữ nguyên; là proposal (safety shield xử lý ở Task 06).
- `momgrap/marl/actions/__init__.py`, `tests_marl/test_codebooks.py` (14 test).

### Tests thêm (14, dùng M=4/L=16/K=2 từ smoke.yaml)
Power: tổng = p_max & floor & shape & finite & nonnegative (mọi index); equal =
uniform; focus đỉnh đúng user; deterministic; reject index lạ; không mutate H/W.
RIS: no-op exact; group action chỉ đổi nhóm đó + wrap; global đổi tất cả; mọi index
hợp lệ + reject; deterministic + không mutate theta. MA: shape (1,M,3) + x giữ
nguyên; no-op exact + các pattern khác biệt; deterministic + không mutate + reject.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_codebooks.py` → **14 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **71 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 06 — Safety shield

**Trạng thái:** hoàn thành. Additive — chỉ thêm `momgrap/marl/actions/safety.py`, không sửa lõi.

### File tạo
- `momgrap/marl/actions/safety.py`:
  - `SafetyProjection` (positions/theta/power_alloc + min_spacing/boundary/power
    violation before&after + repaired + fallback_used).
  - `project_to_safety(positions, theta, power_alloc, cfg, previous_positions=None,
    ris_phase_bits=None)`:
    - **MA**: `repair_positions` rồi **validate** (min spacing ≥ d_min & trong cell);
      nếu vẫn vi phạm → fallback về `previous_positions` (nếu khả thi) hoặc USPA
      centers; `fallback_used=True`. Không tăng `cfg.repair_iters`.
    - **RIS**: wrap `[0,2π)` + quantization bits tùy chọn + kiểm tra finite.
    - **Power**: clamp âm, normalize tổng = p_max, total≈0 → equal; giữ shape (1,1,K).
    - Violation **trước repair** được ghi lại cho reward (Task 10); không mutate input.
- `tests_marl/test_safety.py` — 10 test.

### File sửa
- `momgrap/marl/actions/__init__.py` — export `project_to_safety`, `SafetyProjection`.
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (10)
Input khả thi đi qua nguyên vẹn (repaired=False); positions ngoài cell được repair
về khả thi (repaired=True, không fallback); **fallback** khi repair thất bại (cell
chồng lấn + antenna trùng nhau) → về previous / centers, feasible; RIS wrap +
quantization (2 bit → bội của π/2); power clamp+normalize (sum=p_max, nonneg) &
all-zero→equal; output finite + không mutate input; before-metrics bắt được vi phạm.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_safety.py` → **10 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **81 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 07 — One-interval physical backend

**Trạng thái:** hoàn thành. Additive — chỉ thêm `momgrap/marl/physics/backend.py`, không sửa lõi.

### File tạo
- `momgrap/marl/physics/backend.py`:
  - `PhysicalStepResult` (true_components, estimated_components, precoder, p_user,
    user_rates, min_user_rate, sum_rate, sensing_per_target,
    min_sensing_illumination_db, outage_count).
  - `evaluate_control_interval(environment, positions, theta, power_action, tau, cfg,
    marl_cfg, generator)` — đúng chuỗi timing bắt buộc:
    true = `build_comm_channel_components` → est = `estimate_channel_components(tau)`
    → **W = `bd_precoding(est.aggregate)`** (dùng ESTIMATE) → p_user =
    `decode_power_action(action, est.aggregate, W)` → **rates =
    `evm_sinr_rate(true.aggregate, W, p_user)`** (dùng TRUTH) → B =
    `beampattern(effective_steering(...), W, p_user)`. Không MC-mean (S=1), không
    dùng `Config.power_policy`, không dùng `evaluate_single_regime`.
  - Sensing dB dùng `_DB_EPS=1e-12` khớp `momgrap.objectives`.
- `tests_marl/test_physical_backend.py` — 8 test.

### File sửa
- `momgrap/marl/physics/__init__.py` — export `evaluate_control_interval`,
  `PhysicalStepResult`.
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (8)
Shapes + finite; τ=0 → estimate = truth; W đổi theo nhiễu CSI; realized rate dùng
TRUE channel (khớp tính lại từ `true.aggregate`, khác nếu dùng estimate); power
action đổi p_user → đổi rates (cùng W); **khớp `evaluate_single_regime` tại τ=0 equal
power trên cả 3 regime** (min_user_rate & sensing dB, sai số ≤1e-3/1e-2); không dùng
`Config.power_policy` (đổi policy → p_user không đổi vì action 0 = equal).

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_physical_backend.py` → **8 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **89 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 08 — Communication graph & message system

**Trạng thái:** hoàn thành. Additive — chỉ thêm `momgrap/marl/communication/`, không sửa lõi.

### File tạo
- `momgrap/marl/communication/graph.py`:
  - `AGENT_NAMES` (= `actions.AGENTS`, một nguồn thứ tự duy nhất).
  - `build_adjacency(graph_type, custom, device)` cho preset `none/full/ring/star_bs/
    custom`; `validate_adjacency` (shape (3,3), bool, diagonal = 0).
  - `destination_from_action(agent_index, action)` (0 no-send, 1/2 = other theo thứ
    tự index, loại self); `action_is_valid` / `valid_action_mask` (no-send luôn valid;
    destination valid theo graph mask).
- `momgrap/marl/communication/messages.py`:
  - `MessageContext` + `bs_message` / `ris_message` / `ma_message` /
    `build_agent_messages` — vector semantic deterministic, pad/truncate đúng
    `message_dim`, chỉ dùng dữ liệu estimated/local (không leak true channel).
  - `message_priority(tau, outage, violation)` = uncertainty + urgency.
- `momgrap/marl/communication/budget.py`:
  - `SendRequest`, `DeliveryLog`, `MessageRouter` — filter graph-invalid → rank
    priority (desc) tie-break sender index (asc) → deliver ≤ budget; delivery delay
    ≥ 1 (message chọn ở t → available ở t+delay); log requested/graph_rejected/
    budget_dropped/delivered.
- `momgrap/marl/communication/__init__.py`, `tests_marl/test_messages.py` (14 test).

### Tests thêm (14)
Presets đúng cạnh (none/full/ring/star_bs) + custom + reject self-loop/unknown;
no-self-message mapping + others order; no-send luôn valid + graph mask;
router: self-message bị graph reject, forbidden edge reject, budget giới hạn,
tie-break deterministic theo sender, no-send, delivery delay (không same-step) +
đúng recipient, reject delay=0; message vector finite + đúng dim + deterministic;
priority theo urgency.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_messages.py` → **14 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **103 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 09 — Local observation builder

**Trạng thái:** hoàn thành. Additive — chỉ thêm `momgrap/marl/observations/` +
`momgrap/marl/normalization.py`, không sửa lõi.

### File tạo
- `momgrap/marl/normalization.py` — `FeatureProjection` (ma trận random seed cố định,
  không redraw, checkpointable) + `RunningMeanStd` (Chan parallel, checkpointable).
- `momgrap/marl/observations/features.py` — `complex_to_features(x, mode="summary",
  projection)` (log-mag stats + per-user norms + optional fixed projection),
  `grouped_phase_sincos`, `min_pairwise_distance`, `onehot`, `message_slot`.
- `momgrap/marl/observations/builder.py`:
  - `observation_sizes(evo_cfg, proj_dim)` / `centralized_state_size(...)` — kích thước
    obs **cố định** theo config.
  - `build_actor_observations(...)` — per-agent từ **delayed estimated** CSI: BS
    (aggregate), RIS (reflected + refl/direct ratio + grouped sin/cos), MA (direct +
    offsets + spacing margin + countdown); + previous OWN action one-hot + delivered
    messages (đúng recipient) + tau (nếu expose) + normalized episode time.
    **Không có tham số true channel** (no-leak theo API).
  - `build_centralized_state(...)` — critic dùng true channel + geometry + all actions
    + regime one-hot + tau + step; không đưa cho actor.
- `momgrap/marl/observations/__init__.py`, `tests_marl/test_observations.py` (11 test).

### Tests thêm (11)
Actor/critic shape khớp size khai báo + finite; size không projection; **actor builder
không có param true**; đổi truth → actor obs không đổi (rebuild cùng est) nhưng critic
đổi; delayed CSI được dùng (đổi est → đổi obs); message chỉ tới đúng recipient;
previous action chỉ đổi obs của chính agent; step time tiến; FeatureProjection
deterministic; RunningMeanStd mean/var/normalize đúng.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_observations.py` → **11 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **114 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 10 — PettingZoo parallel environment

**Trạng thái:** hoàn thành. Additive — chỉ thêm `momgrap/marl/envs/` +
`scripts_marl/smoke_env.py`, không sửa lõi.

**Dependencies:** đã `pip install gymnasium==1.3.0 pettingzoo==1.26.1` (được người dùng
cho phép).

### File tạo
- `momgrap/marl/envs/reward.py` — `ObjectiveVector` + `compute_reward` (scalar hóa
  tuyến tính, normalize tường minh từ `RewardConfig`, không dùng thống kê tương lai).
- `momgrap/marl/envs/ramp_parallel_env.py` — `MOMGRAPParallelEnv(ParallelEnv)`:
  - 3 agent, `MultiDiscrete([n_phys, 3])` (thành phần 2 = destination/no-send).
  - `step()` đúng thứ tự: validate → decode → ma_update_period → safety → sample tau →
    control interval (BD từ estimate, rate trên truth, sensing) → messages → graph/
    budget → reward scalar chia sẻ → update config → advance dynamics → append CSI →
    expose messages cho obs kế → truncation ở episode_length.
  - Actor obs chỉ từ **delayed estimate** (không có input true); `state()` centralized
    dùng true channel (không đưa cho actor). Numerical failure → `RuntimeError`.
    Infos đầy đủ (regime/tau/rates/sensing/messages/violations/actions/objective).
- `momgrap/marl/envs/wrappers.py` — `TorchParallelWrapper` (trả torch tensor + state
  torch cho trainer).
- `momgrap/marl/envs/__init__.py`, `scripts_marl/smoke_env.py`,
  `tests_marl/test_parallel_env.py` (13 test).

### Tests thêm (13)
PettingZoo `parallel_api_test`; obs trong observation_space; reset deterministic;
random rollout; **1000 bước finite**; reward chia sẻ giống nhau; **MA update period**
(vị trí chỉ đổi ở bội số period); **message delay + đúng recipient**; **full vs none
graph** (delivered/graph_rejected); tau=0 & tau cao; truncation đúng ở episode_length;
**actor dùng estimate không dùng truth** (rebuild obs từ delayed estimate == obs env);
TorchParallelWrapper.

### Lệnh đã chạy — kết quả
- `python scripts_marl/smoke_env.py` → **ENV SMOKE OK** (parallel_api_test pass,
  obs bs=31/ris=44/ma=43, state=91, rollout 16 bước).
- `python -m pytest -q tests_marl/test_parallel_env.py` → **13 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **127 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 11 — Baseline policies & evaluation pipeline

**Trạng thái:** hoàn thành. Additive — thêm `momgrap/marl/policies/`,
`momgrap/marl/evaluation/`, `scripts_marl/evaluate_baseline.py`; thêm method
`current_estimate()` vào env (file marl); không sửa file lõi.

### File tạo
- `momgrap/marl/policies/baselines.py` — 7 baseline: `RandomPolicy`, `NoOpPolicy`,
  `EqualPowerNoCommunicationPolicy`, `HeuristicFullCommunicationPolicy`,
  `HeuristicNoCommunicationPolicy`, `RandomCommunicationMatchedBudgetPolicy`,
  `StaticRAMPInitializationPolicy` + `make_policy`/`BASELINE_REGISTRY`.
  - Heuristic chỉ dùng **estimated/local** (`env.current_estimate()`): BS equal vs
    inv-gain theo độ chênh gain ước lượng; RIS chấm điểm {no-op, global ±Δ} bằng
    `direct + e^{±jΔ}·reflected` (chỉ từ estimate, không rebuild/true); MA no-op
    off-cycle, expand khi biến thiên không gian direct cao (tôn trọng update period).
- `momgrap/marl/evaluation/metrics.py` — `EpisodeResult` + `summarize_episodes`
  (mean/std/CI95/min/max) + `collect_metadata` (git commit/dirty, Python, Torch,
  CUDA, GPU, seeds).
- `momgrap/marl/evaluation/rollout.py` — `run_episode` / `run_evaluation`.
- `scripts_marl/evaluate_baseline.py` — ghi `outputs_marl/eval/<run_id>/`
  {config_resolved.yaml, metadata.json, episodes.csv, steps.csv, summary.json};
  không đụng `results/`/`figures/`.
- `momgrap/marl/policies/__init__.py`, `momgrap/marl/evaluation/__init__.py`,
  `tests_marl/test_rollout.py` (22 test).

### File sửa
- `momgrap/marl/envs/ramp_parallel_env.py` — thêm `current_estimate()` (accessor
  delayed estimate cho heuristic; không dùng truth).
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (22)
Mỗi baseline chạy hết episode + EpisodeResult finite (parametrized ×7); action hợp lệ
(×7); deterministic reproducible (noop/heuristic/random); no-comm & equal-power gửi 0
message; full-comm gửi >0; noop & static-ramp reconfiguration = 0; run_evaluation +
summarize (n=3, CI); heuristic chạy ở tau cao vẫn finite.

### Lệnh đã chạy — kết quả
- `python scripts_marl/evaluate_baseline.py --policy heuristic_full_comm --seeds 0 1 2`
  → 5 file output đầy đủ; episode_return ≈ 25.6 ± 4.3, messages/step = 1.0.
- `python -m pytest -q tests_marl/test_rollout.py` → **22 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **149 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 12 — Tái sử dụng RAMP/MOMGRAP offline archive

**Trạng thái:** hoàn thành. Additive — thêm `initialization.py` +
`export_ramp_initializations.py`; mở rộng env (file marl); không sửa file lõi.

### File tạo
- `momgrap/marl/physics/initialization.py`:
  - `build_ramp_initializations(cfg, seed, s_eval, n_spaced)` — chạy `run_mfea` →
    `archive.unified_robust_front()` → chọn knee / comm-extreme / sensing-extreme +
    evenly-spaced → decode genotype → `RampInitialization`.
    **Không** đọc genotype từ `results/experiment.pkl`.
  - `select_knee` (max perpendicular distance to chord), `select_max_comm/sensing`,
    `select_evenly_spaced`.
  - `save_initialization`/`load_initialization` (.pt + .json chứa physical_config,
    seed, front, genotypes, selected_indices, positions, theta), `check_compatibility`
    (reject sai M/L/K/Q), `initial_config_from(mode)`.
- `scripts_marl/export_ramp_initializations.py` — ghi
  `outputs_marl/initializations/<name>.{pt,json}` (không đụng `results/`/`figures/`).
- `tests_marl/test_initialization.py` — 8 test.

### File sửa
- `momgrap/marl/envs/ramp_parallel_env.py` — thêm param `ramp_init` + init modes
  `uspa_zero_phase` (đổi tên từ `uspa_zero`), `random`, `ramp_knee`,
  `ramp_comm_extreme`, `ramp_sensing_extreme`, `ramp_random_front_point`.
- `momgrap/marl/physics/__init__.py` — export API initialization.
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (8)
Build shapes + extremes đúng; selection deterministic (front + indices + positions);
config decoded feasible (spacing + cell bounds); save/load round-trip; compatibility
OK + reject sai M/L; `initial_config_from` mọi ramp mode (shape + knee≠comm khác nhau);
env dùng `ramp_knee` init đúng vị trí; env ramp mode thiếu ramp_init → raise.

### Lệnh đã chạy — kết quả
- `python scripts_marl/export_ramp_initializations.py --config configs_marl/smoke.yaml
  --seed 0` → front 4 điểm; comm_extreme max F_com, sensing_extreme max F_sen; .pt+.json.
- `python -m pytest -q tests_marl/test_initialization.py` → **8 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **157 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 13 — Recurrent MAPPO networks

**Trạng thái:** hoàn thành. Additive — chỉ thêm `momgrap/marl/algorithms/`, không sửa lõi.

### File tạo
- `momgrap/marl/algorithms/rmappo/networks.py`:
  - `RNNLayer` — GRU có mask reset, chạy được single-step (rollout) và sequence-chunk
    (training), áp mask mỗi bước để clear hidden ở biên episode.
  - `RecurrentActor` (per agent, không chia sẻ tham số): encoder MLP → GRU → 2 head
    categorical (physical, communication). Hỗ trợ action mask (phys + comm), hidden
    reset, `act` (sample/deterministic), `evaluate_actions` (logprob + entropy).
    no-send luôn valid theo mask từ graph.
  - `CentralizedCritic`: state → MLP → GRU → scalar value.
  - Init orthogonal (gain √2), output head gain nhỏ (0.01); `build_actors`/`build_critic`.
- `momgrap/marl/algorithms/__init__.py`, `momgrap/marl/algorithms/rmappo/__init__.py`,
  `tests_marl/test_networks.py` (10 test + 1 cuda).

### File sửa
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (11)
Actor forward shapes; act shapes + ranges hợp lệ; deterministic = argmax; **comm mask
+ no-send luôn valid**; phys mask; **hidden reset mask** (mask=0 ≡ hidden zeros);
evaluate_actions sequence (logprob/entropy finite); critic forward; **backward pass
finite** (grads); test cuda (marker, chạy trên GPU).

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_networks.py` → **10 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **166 passed** (1 cuda deselected).
- `python -m pytest -q tests_marl/test_networks.py -m cuda` → **1 passed** (GPU).
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 14 — Recurrent rollout buffer & GAE

**Trạng thái:** hoàn thành. Additive — chỉ thêm `buffer.py`, không sửa lõi.

### File tạo
- `momgrap/marl/algorithms/rmappo/buffer.py`:
  - `compute_gae(rewards, values, bootstrap_value, terminated, truncated, gamma, lam,
    terminal_values=None)` — GAE xử lý **truncation tường minh**: bootstrap dùng
    `1-terminated` (truncation vẫn bootstrap terminal value), lan truyền advantage dừng
    ở terminated **hoặc** truncated.
  - `RecurrentRolloutBuffer` — lưu (T,N): obs per-agent, centralized state, phys/comm
    actions, action masks, log-probs, values, shared rewards, terminated/truncated,
    recurrent hidden (actor+critic), reset mask; `insert`/`compute_returns`/
    `normalized_advantages`.
  - `recurrent_minibatches(num_minibatches, seq_len)` — **sequence chunks** (shuffle
    theo chunk, không shuffle timestep trong chunk), giữ hidden ở đầu chunk, pad tail +
    valid mask; flatten time-major khớp `RNNLayer` sequence path.
- `tests_marl/test_buffer.py` — 9 test.

### File sửa
- `momgrap/marl/algorithms/rmappo/__init__.py` — export `RecurrentRolloutBuffer`,
  `compute_gae`.
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (9)
GAE **hand-computed** (γ=λ=1); termination dừng bootstrap+propagation; **truncation dùng
terminal value**; multi-env shape; buffer fill + shapes; minibatch shapes + hidden ở đầu
chunk; hidden khớp chunk start; **padding khi T không chia hết seq_len** (valid = T·N);
**không shuffle timestep trong chunk** (obs time-major theo thứ tự 0,1,2,3).

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_buffer.py` → **9 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **175 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 15 — RMAPPO trainer

**Trạng thái:** hoàn thành. Additive — thêm `trainer.py`/`checkpoint.py` +
`train_rmappo.py`/`evaluate_checkpoint.py`; không sửa lõi.

**Dependencies:** đã `pip install tensorboard==2.21.0` (được người dùng cho phép);
TensorBoard là optional (import graceful), CSV luôn ghi.

### File tạo
- `momgrap/marl/algorithms/rmappo/trainer.py` — `RMAPPOTrainer`:
  synchronous vectorized `num_envs` env; rollout collection; GAE (bootstrap terminal
  value ở truncation); PPO clip + value-clip + entropy; grad clipping; recurrent
  sequence-chunk minibatch; obs/state running normalization (`RunningMeanStd`);
  linear LR anneal (`LambdaLR`); `evaluate` (deterministic); logging TensorBoard+CSV;
  **NaN/Inf policy** (log metadata + emergency checkpoint + raise, không tiếp tục).
- `momgrap/marl/algorithms/rmappo/checkpoint.py` — `save/load_checkpoint`: actors,
  critic, optimizer, scheduler, global_step, update_index, resolved config,
  normalization stats, fixed projection matrix, Python/NumPy/Torch(CPU+CUDA) RNG.
- `scripts_marl/train_rmappo.py` — train → checkpoint → reload → evaluate (outputs
  `outputs_marl/train/<run_id>/`, không đụng `results/`/`figures/`).
- `scripts_marl/evaluate_checkpoint.py` — load checkpoint → evaluate.
- `tests_marl/test_training_smoke.py` — 4 test (gồm toy bandit env).

### File sửa
- `momgrap/marl/algorithms/rmappo/__init__.py` — export trainer + checkpoint.
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (4)
Real-env train → checkpoint → reload (param khớp) → evaluate finite + `train.csv` tồn
tại; checkpoint restore RNG + optimizer + obs_rms (seed khác vẫn khớp);
**PPO cải thiện trên toy bandit** (reward=1 khi bs chọn target≠0; after > before+1 và
≥0.7·max); **NaN policy raises** (critic value_head=nan → update raise RuntimeError).

### Lệnh đã chạy — kết quả
- `python scripts_marl/train_rmappo.py --config configs_marl/smoke.yaml
  --total-timesteps 128` → train 4 updates, checkpoint+TB+CSV, reload, evaluate.
- `python scripts_marl/evaluate_checkpoint.py --checkpoint <...>` → khớp kết quả reload.
- `python -m pytest -q tests_marl/test_training_smoke.py` → **4 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **179 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 16 — Learned selective communication

**Trạng thái:** hoàn thành. Additive — thêm file marl + mở rộng env/trainer/policies;
không sửa lõi. (Cơ chế comm head/mask/graph/budget/message-cost/delay đã có từ Task
08/10/13/15; Task 16 thêm forced modes, curriculum, metrics, ablation.)

### File tạo
- `momgrap/marl/communication/modes.py` — `COMM_MODES` (none/full/random/learned),
  `comm_mask_for_mode` (training-consistent: none→chỉ no-send, full→tắt no-send,
  learned→graph mask), `forced_comm_action` (env forcing cho eval), `CommunicationCurriculum`
  (anneal message-cost, tránh collapse không cần bonus tùy tiện).
- `momgrap/marl/evaluation/comm_metrics.py` — `CommStats`, `comm_stats_from_steps`
  (send-request/delivery/drop/graph-reject rate, messages/step), `evaluate_comm_modes`
  (ablation 4 mode + message_utility vs none + performance-per-delivered-message).
- `tests_marl/test_communication_modes.py` — 10 test.

### File sửa
- `momgrap/marl/envs/ramp_parallel_env.py` — param `comm_mode` +
  `set_communication_mode`/`set_message_cost`; forcing comm trong step; reward dùng
  message-cost override (dataclasses.replace).
- `momgrap/marl/algorithms/rmappo/trainer.py` — param `comm_mode` (comm mask theo mode)
  + `train(curriculum=)` hook (set_message_cost mỗi update).
- `momgrap/marl/policies/baselines.py` — `TrainedActorPolicy` (bridge actors đã train
  vào rollout/ablation).
- `communication/__init__.py`, `evaluation/__init__.py`, `policies/__init__.py`,
  `CHANGELOG_EVOCOMM.md`.

### Tests thêm (10)
`forced_comm_action` + `comm_mask_for_mode` + `CommunicationCurriculum.cost_at`; env
full/none forcing; message-cost override giảm reward; **ablation 4 mode** (none=0 msg,
full>0, utility); trainer comm_mode none→0 msg / full→>0 msg; `TrainedActorPolicy` chạy.

### Ghi chú
- Curriculum truyền **programmatically** (không thêm field vào `EvoCommConfig` đã đóng
  băng ở Task 01) — script/trainer dựng `CommunicationCurriculum` và truyền vào `train`.
- Training dùng comm **mask** (PPO-consistent); eval dùng env **forcing** (đo cùng policy
  dưới mọi mode). Không thêm communication bonus tùy tiện.

### Lệnh đã chạy — kết quả
- `python -m pytest -q tests_marl/test_communication_modes.py` → **10 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **189 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 17 — Evolutionary communication-graph search

**Trạng thái:** hoàn thành. Additive — thêm `momgrap/marl/evolution/` + script; mở rộng
env (`set_adjacency`); không sửa lõi.

### File tạo
- `momgrap/marl/evolution/graph_encoding.py` — 6-bit genotype (bs→ris, bs→ma, ris→bs,
  ris→ma, ma→bs, ma→ris); `encode_graph`/`decode_graph`; `GraphConstraints` +
  `repair_graph` (max_edges, allow_empty, weak_connectivity, bs_incident).
- `momgrap/marl/evolution/graph_fitness.py` — `graph_objectives` (F1 min-rate, F2
  sensing, F3 -messages, F4 -outage; **common random numbers** trên cùng seed/tau/
  regime/episode), `FitnessCache` (keyed by prefix policy+config+seeds+taus + bitstring),
  `make_cached_evaluator`.
- `momgrap/marl/evolution/graph_search.py` — `run_graph_search` (NSGA-II: binary
  tournament + uniform crossover + bit-flip + repair; survival =
  `environmental_selection`; **không dùng `hypervolume_2d`**), `evaluate_preset_graphs`
  (none/full/ring/star_bs + random).
- `scripts_marl/evolve_comm_graph.py` — chạy search (baseline hoặc trained checkpoint),
  ghi `outputs_marl/graph_search/<run_id>/` {pareto_graphs.json, pareto_objectives.csv,
  population_history.csv, cache.pkl, config_resolved.yaml, metadata.json}.
- `tests_marl/test_graph_search.py` — 12 test.

### File sửa
- `momgrap/marl/envs/ramp_parallel_env.py` — `set_adjacency` (đổi graph, rebuild router).
- `CHANGELOG_EVOCOMM.md`.

### Tests thêm (12)
encode/decode round-trip + edge mapping; repair (max_edges/non-empty/bs-incident/weak-
connectivity/deterministic); `graph_objectives` shape 4 + **CRN deterministic**; graph
ảnh hưởng comm (none F3=0, full F3<0); FitnessCache put/get + save/load; cached evaluator
reuse; `run_graph_search` (pareto≥1, pop giữ nguyên, history=max_gen, objectives finite);
`evaluate_preset_graphs` (none/full/ring/star_bs + random).

### Lệnh đã chạy — kết quả
- `python scripts_marl/evolve_comm_graph.py --config smoke.yaml --pop-size 8 --max-gen 3`
  → pareto front + baselines + 6 file output.
- `python -m pytest -q tests_marl/test_graph_search.py` → **12 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **201 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 18 — Scientific evaluation

**Trạng thái:** hoàn thành. Additive — thêm `statistics.py` + `run_ablation.py` +
`eval.yaml`; không sửa lõi.

### File tạo
- `momgrap/marl/evaluation/statistics.py` — `mean_ci` (t-based 95% CI), `paired_t_test`
  (`ttest_rel`), `cohens_d_paired`, `holm_correction` (step-down), `pareto_front`
  (m-obj, maximize mask), `pareto_coverage` (C-metric), `epsilon_indicator` (additive),
  `pairwise_2d_projections`. **Không** tính 3D HV (không mis-adapt `hypervolume_2d`).
- `scripts_marl/run_ablation.py` — quét lưới (regime × tau × delay × budget × comm-method
  × init), eval trên **paired seeds**, per-cell mean±CI + runtime + GPU mem; **paired
  comparisons vs none + Holm + Cohen's d**; **Pareto** (min-rate↑, sensing↑, messages↓);
  ghi `outputs_marl/ablation/<run_id>/` {cells.csv, per_episode.csv, comparisons.json,
  pareto.json, config_resolved.yaml, metadata.json}.
- `configs_marl/eval.yaml` — lưới eval nhỏ mặc định.
- `tests_marl/test_statistics.py` — 5 test.

### File sửa
- `momgrap/marl/evaluation/__init__.py` — export statistics; `CHANGELOG_EVOCOMM.md`.

### Tests thêm (5)
mean_ci; paired t-test + Cohen's d (+ n=1/identical edge cases); Holm (monotone,
significant); pareto_front (maximize + mixed); coverage + epsilon indicator.

### Ghi chú
- Memory axis (feed-forward vs recurrent): actor hiện là recurrent — feed-forward là
  giới hạn được ghi nhận (chưa triển khai actor FF riêng).
- Budget sweep đổi kích thước obs (message slot) → policy đã train chỉ eval được ở đúng
  budget lúc train; baseline policy chạy mọi budget. EA-graph method cần adjacency truyền vào.

### Lệnh đã chạy — kết quả
- `python scripts_marl/run_ablation.py --eval-config configs_marl/eval.yaml`
  → 24 cells × 3 seeds; comparisons + pareto + 6 file output.
- `python -m pytest -q tests_marl/test_statistics.py` → **5 passed**.
- `python -m pytest -q -m "not slow and not cuda"` → **206 passed**.
- Không chạm file lõi → legacy regression test trong suite (qua).

---

## Task 19 — Documentation & CI

**Trạng thái:** hoàn thành. Additive — chỉ thêm docs + workflow, không sửa lõi.

### File tạo
- `README_EVOCOMM.md` — kế thừa từ momgrap / cái mới / install / legacy smoke / MARL
  smoke / baseline eval / training / checkpoint eval / RAMP init / graph search /
  ablation / output folders / reproducibility / **limitations**.
- `docs/EVOCOMM_ENVIRONMENT_SPEC.md`, `EVOCOMM_OBSERVATIONS.md`, `EVOCOMM_ACTIONS.md`,
  `EVOCOMM_REWARD.md`, `EVOCOMM_EXPERIMENT_PROTOCOL.md`.
- `.github/workflows/evocomm-tests.yml` — job `cpu-tests` (CPU torch + requirements-marl
  → `pytest -q tests_marl -m "not slow and not cuda"` → smoke_backend → smoke_env);
  job `legacy-smoke` (Stage A/B/C `smoke_test.py`) chỉ chạy `workflow_dispatch` (manual,
  tách riêng để CI chính nhanh & không tái tạo figures resubmission).

### File sửa
- `CHANGELOG_EVOCOMM.md`.

### Lệnh CI đã xác minh (local)
- `python -m pytest -q tests_marl -m "not slow and not cuda"` → **206 passed** (1 cuda deselected).
- `python scripts_marl/smoke_backend.py` → **BACKEND SMOKE OK**.
- `python scripts_marl/smoke_env.py` → **ENV SMOKE OK**.
- Không chạm file lõi → legacy regression test trong suite (qua).
