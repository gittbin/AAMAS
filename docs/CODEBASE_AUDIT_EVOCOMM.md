# CODEBASE AUDIT — EvoComm-MOMGRAP (Task 00 snapshot)

Bản chụp trạng thái codebase `AAMAS_new / momgrap` **trước khi** thêm bất kỳ mã MARL
nào. Mục đích: khóa hành vi hiện tại làm mốc hồi quy (regression guard) cho các task
sau. Task 00 **không** sửa bất kỳ file `momgrap/*.py` nào — chỉ ghi lại hiện trạng.

---

## 1. Snapshot môi trường

| Mục | Giá trị |
| --- | --- |
| Commit hash | `1ba133df7ef7b8612311323fcc6cb8e1bec92b37` |
| Branch | `main` |
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA available | `True` |
| CUDA version (torch) | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Ngày chụp | 2026-08-04 |

### `git status --short` tại thời điểm chụp

```text
 M figures/fig1_unified_front.png
 M figures/fig2_hv_over_gen.png
 M figures/fig3_hv_cdf.png
 M momgrap/baselines.py
 M momgrap/comm.py
 M momgrap/config.py
 M momgrap/mfea.py
 M momgrap/objectives.py
 M momgrap/plots.py
 M momgrap/sensing.py
 M results/experiment.pkl
 M results/run.log
?? figures_paper/
?? resubmission/
```

> **Cảnh báo hồi quy:** working tree đang bẩn với các thay đổi resubmission quan
> trọng. Không được `git reset --hard`, `git checkout -- .`, `git restore .`,
> `git clean -fd(x)`, và không ghi đè `results/`, `figures/`, `figures_paper/`,
> `figures_diag/`, `resubmission/`.

### Commit gần đây

```text
1ba133d seed_table.py: single-seed Table-I breakdown (HV@seed vs aggregate, knee, IGD)
89adb46 fig1: overlay table methods (RAMP emphasised); make_fig1_seed.py renders Fig 1 for a chosen seed
157bd5f add_seeds.py: append more seeds to an experiment pickle (no full re-run)
d64ba5e per_seed_report.py: per-seed final-HV breakdown + paired diffs + CSV
519f278 fig1: do NOT overlay real-time envelope (misleading on the robust-front plane)
```

---

## 2. Module map (`momgrap/`)

| Module | Vai trò |
| --- | --- |
| `__init__.py` | Export `Config`, `default_config`, `smoke_config`. |
| `config.py` | Dataclass `Config` (mọi hyperparameter), `RegimeSpec`, `default_config`, `smoke_config`. |
| `channels.py` | Geometry (BS/RIS grids, steering), `Environment` (MC snapshot), `sample_environment(s)`, `build_comm_channels`, `build_G_geo`. |
| `comm.py` | `bd_precoding`, `power_allocation`, `evm_sinr_rate`. |
| `sensing.py` | `effective_steering`, `beampattern` (sensing *illumination* / beampattern quality). |
| `genotype.py` | `clip_genotype`, `repair_positions`, `decode`. |
| `objectives.py` | `evaluate_single_regime`, `robust_evaluate`, `evaluate_population`, draw helpers. |
| `operators.py` | Biến đổi tiến hóa (SBX/linear crossover, mutation). |
| `nsga.py` | `fast_nondominated_sort` (m objectives), `crowding_distance`, `environmental_selection`. |
| `rmp.py` | Adaptive RMP controller. |
| `mfea.py` | `MFEAResult`, `run_mfea` (MO-MFEA driver). |
| `archive.py` | `Archive` (rank-1 genotypes), `unified_robust_front`, `robust_objectives`. |
| `metrics.py` | `hypervolume_2d` (**chỉ đúng cho 2 objectives**), HV reference, trackers. |
| `baselines.py` | `run_adaptive` và các baseline so sánh. |
| `experiments.py` | `run_all` — pipeline thí nghiệm (Stage C). |
| `diagnostic.py` | Diagnostic protocol (synergy↔conflict sweep). |
| `plots.py` | `make_all_figures` và các hàm vẽ. |
| `tables.py` | Sinh bảng số liệu. |
| `logging_utils.py` | Logger tiện ích. |

---

## 3. `Config` defaults hiện tại (mô hình vật lý)

Nguồn: `momgrap/config.py` (`Config`, `default_regimes`, `smoke_config`).

| Trường | Default | Ghi chú |
| --- | --- | --- |
| `M` | 16 | movable antennas |
| `L` | 64 | RIS elements |
| `K` | 4 | users |
| `Q` | 3 | sensing targets |
| `carrier_hz` | 28e9 | 28 GHz |
| `bs_grid_rows` × `bs_grid_cols` | 4 × 4 | = M |
| `ris_grid_rows` × `ris_grid_cols` | 8 × 8 | = L |
| `cell_spacing_factor` | 1.0 | × λ |
| `cell_extent_factor` | 0.5 | × λ (nửa extent = 0.25λ) |
| `ris_spacing_factor` | 0.5 | × λ |
| `ris_position` | (30, 40, 10) m | |
| `user_radius_min` / `max` | 50 / 200 m | |
| `user_sector_center_deg` | 0.0 | |
| `user_sector_width_deg` | 60.0 | |
| `target_sep_deg` (Δφ) | 20.0 | within-task offset |
| `target_cluster_width_deg` | 20.0 | |
| `delta_task_deg` | 0.0 | between-task |
| `direct_atten_db` | 0.0 | |
| `p_max_dbm` | 30.0 | tổng công suất BS |
| `bandwidth_hz` | 180e3 | |
| `noise_figure_db` | 9.0 | |
| `evm_kappa` | 0.05 | |
| `pl0_db` / `pl_exponent` | 61.4 / 2.2 | |
| `n_scatter_paths` | 4 | |
| `scatter_sector_deg` | 120.0 | |
| `gamma_ris` | 1.0 | sức mạnh path phản xạ sensing |
| `softclip_mu` | 2.0 | |
| `repair_iters` | 3 | |
| `d_min_factor` | 0.5 | d_min = 0.5λ |
| `regimes` | LoS / Rayleigh / Rician(6 dB) | `default_regimes()` |
| `shared_geometry` | True | |
| `pop_size` | 200 | |
| `max_gen` | 300 | |
| `mutation_sigma` | 0.1 | |
| `sbx_eta` | 15.0 | |
| `crossover_kind` | "sbx" | |
| `n_seeds` | 20 | |
| `mc_samples` | 32 | S |
| `rmp_init/min/max` | 0.3 / 0.05 / 0.5 | |
| `rmp_eta` | 0.05 | |
| `rmp_signal` | "relative" | {relative, absolute} |
| `survival_mode` | "rank1" | {rank1, hv_contrib} |
| `power_policy` | **"equal"** | {equal, inv_gain} |
| `cross_regime_agg` | "min" | worst-regime |
| `hv_ref_com` | 0.0 | |
| `hv_ref_sen_db` | -30.0 | |
| `seed` | 0 | |
| `device_str` | "auto" | {auto, cpu, cuda} |
| `dtype_real` | torch.float32 | |
| `paired_envs` | True | |
| `eval_batch_size` | 256 | |
| `archive_max_size` | 20000 | |
| `log_every` | 25 | |

### Thuộc tính dẫn xuất

| Property | Công thức |
| --- | --- |
| `D` | `2*M + L` (= 144 với default; = 40 với smoke M=4/L=16/... — xem dưới) |
| `T` | `len(regimes)` (= 3) |
| `wavelength` | `C_LIGHT / carrier_hz` |
| `k0` | `2π / λ` |
| `d_min` | `d_min_factor * λ` |
| `dtype_complex` | complex64 (nếu real=float32) |
| `p_max_watt` | `10^(p_max_dbm/10)/1000` |
| `p_per_user` | `p_max_watt / K` |
| `noise_power_watt` | từ N0 + bandwidth + noise figure |
| `device` | cuda nếu auto & available, ngược lại cpu |

### `smoke_config()` overrides mặc định

```python
pop_size=24, max_gen=8, mc_samples=4, n_seeds=1
# (giữ nguyên geometry vật lý mặc định: M=16, L=64, K=4, Q=3 -> D=144)
```

### `Config.validate()` ràng buộc

- `bs_grid_rows * bs_grid_cols == M`
- `ris_grid_rows * ris_grid_cols == L`
- `crossover_kind ∈ {sbx, linear}`
- `survival_mode ∈ {rank1, hv_contrib}`
- `rmp_signal ∈ {relative, absolute}`
- `power_policy ∈ {equal, inv_gain}`
- `cross_regime_agg == "min"`

---

## 4. Chữ ký hàm public (khóa lại, không được thay đổi semantics)

Genotype: `x ∈ [0,1]^D`, `D = 2M + L`. `2M` gene đầu = MA offsets, `L` gene cuối = RIS phases.

### 4.1. `momgrap.channels`

```python
class Environment:
    regime: RegimeSpec
    S: int
    user_pos: Tensor        # (S,K,3)
    user_dir: Tensor        # (S,K,3) unit BS->user
    user_amp: Tensor        # (S,K) sqrt path loss BS->user
    scat_dir_bu: Tensor     # (S,K,P,3) NLoS arrival dirs (BS->user)
    scat_gain_bu: Tensor    # (S,K,P) complex
    G_nlos: Tensor          # (S,L,M) complex iid
    g_los_dir: Tensor       # (S,K,3) unit RIS->user
    g_amp: Tensor           # (S,K) sqrt path loss RIS->user
    g_nlos: Tensor          # (S,K,L) complex iid
    sector_center_deg: float = 0.0

def sample_environment(
    regime: RegimeSpec, S: int, generator: torch.Generator, cfg: Config,
    sector_center_deg: float | None = None,
) -> Environment

def build_comm_channels(
    P_ant: torch.Tensor, theta: torch.Tensor, env: Environment, cfg: Config,
) -> torch.Tensor
# trả Hh = h_k^H, shape (N,S,K,M) complex. Hh = d + reflected (direct + RIS-reflected).
```

### 4.2. `momgrap.comm`

```python
def bd_precoding(Hh: torch.Tensor, cfg: Config) -> torch.Tensor
# W (N,S,M,K) complex, cột unit-norm, null-space BD.

def power_allocation(Hh: torch.Tensor, W: torch.Tensor, cfg: Config) -> torch.Tensor | None
# "equal" -> None (caller dùng scalar P_max/K); "inv_gain" -> (N,S,K) tổng = P_max.

def evm_sinr_rate(
    Hh: torch.Tensor, W: torch.Tensor, cfg: Config, p_user: torch.Tensor | None = None,
) -> torch.Tensor
# rate (N,S,K) real (bps/Hz). p_user=(N,S,K) hoặc None (uniform).
```

### 4.3. `momgrap.sensing`

```python
def effective_steering(
    P_ant: torch.Tensor, theta: torch.Tensor, cfg: Config,
    sector_center_deg: float | None = None,
) -> torch.Tensor
# a_eff (N,Q,M) complex (deterministic).

def beampattern(
    a_eff: torch.Tensor, W: torch.Tensor, cfg: Config, p_user: torch.Tensor | None = None,
) -> torch.Tensor
# B (N,S,Q) real >= 0. Đây là sensing ILLUMINATION / beampattern quality,
# KHÔNG phải detection probability / CRB / Fisher / sensing SINR.
```

### 4.4. `momgrap.genotype`

```python
def decode(genotype: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]
# -> (P_ant (N,M,3), theta (N,L)); đã gọi repair_positions() bên trong.
```

### 4.5. `momgrap.objectives`

```python
def evaluate_single_regime(
    genotypes: torch.Tensor, env: Environment, cfg: Config,
) -> torch.Tensor
# F (N,2): col0 = F_com (bps/Hz), col1 = F_sen (dB). Cả hai maximize.
```

### 4.6. `momgrap.mfea`

```python
def run_mfea(cfg: Config, seed: int | None = None, tag: str = "") -> MFEAResult
# MFEAResult có: rmp_history, rho_history, hv_history, final_pop, final_skill,
#   per_regime_front (dict t -> (front (P,2), genos (P,D))), archive.

# Archive.unified_robust_front(generator, S_eval=None) -> (front (P,2), genos (P,D))
```

---

## 5. Ràng buộc tái sử dụng (cho các task sau)

- **Tái sử dụng nguyên vẹn:** `fast_nondominated_sort` (m objectives),
  `crowding_distance`, `environmental_selection`, `operators`, `Archive`, `run_mfea`.
- **Không dùng cho >2 objectives:** `metrics.hypervolume_2d` chỉ đúng cho đúng 2
  objectives → không dùng cho Pareto front 3D/4D của communication graph.
- **`Environment` là MC snapshot, không phải sequential env** — không có
  `reset()`/`step()`/velocity/temporal correlation. Không gọi `sample_environment()`
  độc lập ở mỗi MARL step (mất tính liên tục theo thời gian).
- **`resubmission/task10_online.py`** là per-realisation re-optimization, **không**
  phải online MARL environment; chỉ dùng làm runtime/reference baseline.
- **`results/experiment.pkl`** chủ yếu lưu objective fronts (NumPy), **không** bảo
  đảm có genotype archive → không dùng làm nguồn genotype cho khởi tạo MA/RIS.

---

## 6. Baseline đã xác minh

`python smoke_test.py` chạy thành công (Stage A/B/C) trên codebase này (đã xác minh
trước Task 00). Mọi thay đổi mới phải giữ lệnh này tiếp tục qua.
