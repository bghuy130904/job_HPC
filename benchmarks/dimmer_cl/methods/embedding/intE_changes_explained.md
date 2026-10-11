# Những thay đổi trong pipeline intE của benchmark dimmer_cl

Ngày lập: 11/10/2026. Cập nhật đường dẫn launcher theo commit `00f744a` của người dùng. Báo cáo so sánh file gốc `intE_obmp2inb3lyp.py` đã gửi với file hiện tại `benchmarks/dimmer_cl/methods/embedding/intE.py` trong repo `bghuy130904/job_HPC`, branch `workspace/benchmark-layout`.

Các số dòng “cũ” thuộc file gốc, các số dòng “mới” thuộc `intE.py` hiện tại (442 dòng). Chúng có thể thay đổi khi file được chỉnh sửa tiếp. Phụ lục cuối là diff đầy đủ của hai file, bao gồm cả thay đổi chú thích và thông báo.

## 1. Chuẩn hóa tham số R: sửa lỗi kiểu dữ liệu trước vòng scan

**Vị trí:** cấu hình R, cũ dòng 31; mới dòng 45–54. Vòng scan gây lỗi ở file cũ dòng 384, hiện tại dòng 400.

Trước, kết quả `json.loads()` được dùng trực tiếp:
```python
ENV_R = os.environ.get("R")
# Mặc định chuỗi scan R nếu không truyền từ môi trường
R_VALUES = json.loads(ENV_R) if ENV_R else [0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50, 2.00]
```
Sau:
```python
ENV_R = os.environ.get("R")
# Mặc định chuỗi scan R nếu không truyền từ môi trường
R_VALUES = json.loads(ENV_R) if ENV_R else CONFIG['R_VALUES']
if isinstance(R_VALUES, (int, float)):
    R_VALUES = [R_VALUES]
if not isinstance(R_VALUES, list) or not R_VALUES:
    raise ValueError("R must be a number or nonempty JSON list")
R_VALUES = [float(r) for r in R_VALUES]
if any(not np.isfinite(r) or r <= 0 for r in R_VALUES):
    raise ValueError("R values must be finite and positive")
```
**Tại sao sửa:** `R='1.0'` được đọc thành số float `1.0`; số đơn không thể dùng trong `enumerate(R_VALUES)`. Lỗi là `TypeError: 'float' object is not iterable`, xảy ra trước khi bắt đầu các phép tính. `R='1.05'`, `R='2.0'` cũng có cùng vấn đề; không riêng R=1.0. `R='[1.0]'` vốn đã là danh sách.

Bản mới bọc số đơn thành danh sách, chuyển từng phần tử sang float và từ chối danh sách rỗng, số không hữu hạn hoặc R ≤ 0. Đây là sửa đầu vào, không chứng minh hay giải quyết nguyên nhân OBMP2 không hội tụ tại một hình học cụ thể. R trong driver là **hệ số R/R0**, không phải khoảng cách Å.

## 2. Chỉ gọi kernel một lần, dùng đúng total embedding energy

**Vị trí:** `run_embedding_calculation()` cũ dòng 231–256 và hàm `_extract_total_energy()` cũ; `_run_embedding_calculation()` mới dòng 205–216.

Trước:
```python
    start_embed = time.time()
    result = None
    
    # Chạy tính toán OBMP2 (thử .run(), nếu không có/lỗi thì chuyển qua .kernel())
    if hasattr(mp_emb, "run"):
        try:
            result = mp_emb.run()
        except Exception as e:
            print(f"[WARNING] mp_emb.run() gặp sự cố: {e}")

    if getattr(mp_emb, "e_corr", None) is None and hasattr(mp_emb, "kernel"):
        print(f"[INFO] Running OBMP2 kernel for {label}...")
        result = mp_emb.kernel()

    # Bổ sung e_corr dự phòng nếu result trả về giá trị
    if getattr(mp_emb, "e_corr", None) is None and result is not None:
        if isinstance(result, (float, int, np.floating)):
            scf_e = getattr(getattr(mp_emb, "_scf", None), "e_tot", 0.0)
            mp_emb.e_corr = float(result) - scf_e
        elif isinstance(result, (tuple, list)) and len(result) > 0:
            if isinstance(result[0], (float, int, np.floating)):
                mp_emb.e_corr = float(result[0])

    print(f"Embedding runtime = {time.time() - start_embed:.3f} s")
    
    e_tot = _extract_total_energy(result, mp_emb)
```
Sau:
```python
    start_embed = time.time()
    e_tot = float(mp_emb.kernel())
    if METHOD == 'obmp2':
        record['OBMP2_converged'] = bool(getattr(mp_emb, 'converged', False))
        record['active_nocc'] = str(tuple(int(n) for n in mp_emb.get_nocc()))
        record['active_nmo'] = str(tuple(int(n) for n in mp_emb.get_nmo()))
    else:
        record['Solver'] = 'one-shot UMP2; embedded UHF convergence checked by shared module'
    if not np.isfinite(e_tot):
        raise RuntimeError(f"{label}: non-finite embedding total energy")
    if METHOD == "obmp2" and not bool(getattr(mp_emb, "converged", False)):
        raise RuntimeError(f"{label}: OBMP2 Not convergence")
```
**Tại sao sửa:** với backend OBMP2_CL đã kiểm tra, `.run()` thực hiện `.kernel()` nhưng trả về object. Backend có thể không đặt `e_corr` theo giả định của driver cũ; điều kiện kiểm tra `e_corr is None` vì vậy gọi `.kernel()` lần thứ hai trên cùng object. Ngoài việc tính lặp, trạng thái backend có thể đã thay đổi sau lần gọi đầu.

Hàm `_extract_total_energy()` và đoạn tự dựng `mp_emb.e_corr = result - scf_e` đã bị bỏ. Backend embedding hiện tại trả trực tiếp total embedding energy từ `kernel()`; driver nhận đúng giá trị đó. Tự cộng năng lượng UHF toàn hệ với một `e_corr` suy đoán không thể thay thế công thức tái tổ hợp embedding của backend.

**Giới hạn tương thích:** driver dựa trên contract `kernel() -> total energy` của backend đã dùng để test. Nếu một phiên bản pyCMF khác trả correlation energy hoặc tuple, cần kiểm tra lại contract; không thể dùng nguyên giả định này cho mọi phiên bản.

## 3. UHF ban đầu: bổ sung điều kiện chấp nhận và thay đổi tham số chạy

**Vị trí:** `_run_embedding_calculation()`, cũ dòng 189–203; mới dòng 154–177.

Trước:
```python
    mol.incore_anyway = True
    
    mf = scf.UHF(mol).density_fit()
    mf.max_memory = MEMORY
    mf.verbose = mol.verbose
    
    start_scf = time.time()
    mf.kernel()
    if not mf.converged:
        print(f"[WARNING] UHF did not converge for {label}.")
    print(f"UHF runtime = {time.time() - start_scf:.3f} s")
    print(f"UHF Energy = {mf.e_tot:.12f} Eh")

    # Kiểm tra và ổn định nghiệm SCF
    mf = stabilize_scf(mf, max_macro_cycles=10, verbose=True)
```
Sau:
```python
    mol.incore_anyway = False

    mf = scf.UHF(mol).density_fit()
    mf.max_memory = MEMORY
    mf.verbose = mol.verbose
    mf.conv_tol = 1e-10
    mf.max_cycle = 200

    start_scf = time.time()
    mf.kernel()
    record['UHF_converged'] = bool(mf.converged)
    if not mf.converged:
        raise RuntimeError(f"{label}: UHF Not convergence")
    print(f"UHF runtime = {time.time() - start_scf:.3f} s")
    print(f"UHF Energy = {mf.e_tot:.12f} Eh")

    # Kiểm tra và ổn định nghiệm SCF
    mf = stabilize_scf(mf, max_macro_cycles=10, verbose=True)
    if not mf.converged:
        raise RuntimeError(f"{label}: stabilized UHF Not convergence")
    _, _, stable, _ = mf.stability(return_status=True)
    record['UHF_stable'] = bool(stable)
    if not stable:
        raise RuntimeError(f"{label}: unstable UHF after stabilization")
```
**Tại sao sửa:** bản cũ chỉ in cảnh báo khi UHF chưa hội tụ rồi vẫn chạy embedding. Bản mới dừng phép tính của fragment đó và ghi trạng thái, tránh nhận năng lượng từ một reference không đạt yêu cầu. Sau `stabilize_scf`, kiểm tra lại hội tụ và **internal stability** qua giá trị `stable` mà lời gọi này nhận được; đây không phải kiểm tra mọi dạng instability có thể có.

**Thay đổi tham số thực tế cần biết:** thêm `mf.conv_tol = 1e-10`, `mf.max_cycle = 200`, và đổi `mol.incore_anyway` từ True thành False. Hai tham số đầu làm điều kiện UHF chặt hơn và cho phép nhiều vòng hơn; có thể ảnh hưởng runtime và nghiệm thu được. False bỏ việc ép tích phân ở trong RAM; không đảm bảo hết OOM vì backend vẫn có thể tạo mảng lớn. Không sửa thuật toán SCF trong source PySCF/pyCMF.

**Hành vi cụ thể:** nếu UHF đầu tiên chưa hội tụ, driver dừng fragment ngay, trước bước ổn định; bản này không tự cứu một reference chưa hội tụ bằng `stabilize_scf`.

## 4. OBMP2: đặt số vòng lặp rõ ràng và kiểm tra cờ hội tụ

**Vị trí:** cấu hình `mp_emb`, cũ dòng 228–229; mới dòng 202–216.

Trước:
```python
    if not hasattr(mp_emb, "niter"):
        mp_emb.niter = 1000
```
Sau, phần tham số:
```python
    mp_emb.niter = int(os.environ.get("OBMP2_NITER", "1000"))
    mp_emb.diis_space = int(os.environ.get("DIIS_SPACE", "8"))
```
**Tại sao sửa:** điều kiện `if not hasattr(..., 'niter')` không đổi giá trị mặc định nếu class đã có thuộc tính `niter`, nên số 1000 có thể chưa bao giờ được áp dụng. Nay đặt trực tiếp, cho phép thay bằng `OBMP2_NITER`; `DIIS_SPACE` mặc định 8.

Sau kernel, driver kiểm tra năng lượng hữu hạn và cờ `mp_emb.converged` cho **OBMP2**; nếu không đạt thì không chấp nhận E_int. Ghi thêm `active_nocc` và `active_nmo` mà object backend cung cấp.

Đây có thể thay đổi số vòng lặp/DIIS và tiêu chuẩn chấp nhận kết quả tại driver. Không sửa biểu thức năng lượng, BCH hoặc điều kiện hội tụ bên trong pyCMF. Giá trị `mp_emb.thresh = 1e-6` đã có trong file gốc và vẫn giữ nguyên.

## 5. Log từng fragment và bảng hội tụ riêng

**Vị trí:** thêm wrapper `run_embedding_calculation()`, mới dòng 226–254; hàm thực thi cũ đổi tên thành `_run_embedding_calculation()`.

Code hiện tại:
```python
def run_embedding_calculation(mol, active_atoms, xc_env='b3lyp', label=""):
    """Call the unchanged backend once; retain fragment diagnostics on failure."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label.replace(' ', '_')}.log"
    record = dict(Method=METHOD, Label=label, Basis=BASIS, Status="FAILED", Log=str(log_path))
    original_stdout = mol.stdout
    print(f"Running {label}; log: {log_path}")
    try:
        with log_path.open('w', buffering=1) as log:
            mol.stdout = log
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                energy = _run_embedding_calculation(mol, active_atoms, xc_env, label, record)
        # The backend does not expose the DFT and embedded-UHF SCF objects.
        # Reject an explicit SCF failure reported in its calculation log.
        backend_log = log_path.read_text().split(f'--- {METHOD.upper()}-in-B3LYP Embedding', 1)[-1]
        backend_scf_failed = 'SCF not converged' in backend_log
        record['Backend_SCF_no_failure_reported'] = not backend_scf_failed
        if backend_scf_failed:
            raise RuntimeError(f"{label}: backend SCF Not convergence; see {log_path}")
        record.update(Status="OK", E_total_Eh=energy)
        return energy
    except Exception as exc:
        record['Status'] = "Not convergence" if any(token in str(exc).lower() for token in ('not converg', 'unstable')) else "FAILED"
        record['Error'] = str(exc)
        raise
    finally:
        mol.stdout = original_stdout
        CONVERGENCE_RECORDS.append(record)
        pd.DataFrame(CONVERGENCE_RECORDS).to_csv(CONVERGENCE_FILE, index=False)
```
**Tại sao sửa:** lưu toàn bộ stdout/stderr Python cùng output PySCF qua `mol.stdout` thành log riêng cho mỗi fragment. `finally` khôi phục stdout và ghi `convergence.csv` kể cả khi có exception. Bảng chứa Method, Label, Basis, Status, Log, các cờ hội tụ đã lấy được, E_total_Eh khi thành công và Error khi thất bại.

**Kiểm tra SCF nội bộ:** vì driver không nhận trực tiếp object full-system DFT và embedded UHF trong backend OBMP2, nó tìm chuỗi `SCF not converged` trong phần log sau marker embedding. Nếu xuất hiện, fragment bị từ chối. `Backend_SCF_no_failure_reported=True` chỉ nghĩa là **không thấy chuỗi báo lỗi đó**, không phải cờ hội tụ đầy đủ của từng SCF nội bộ. Backend dùng cách báo lỗi khác có thể không được nhận diện bằng bộ lọc này.

Phân loại exception dùng `str(exc).lower()` với token `not converg` hoặc `unstable`; nhận cả “Not convergence”, “not converged” và “did not converge”. Lỗi khác được ghi FAILED.

## 6. Một fragment lỗi vẫn tính hai fragment còn lại; không điền số 0

**Vị trí:** `calculate_interaction_energy_at_r()`, cũ gọi tuần tự không bắt lỗi từng fragment; mới dòng 333–360.

Code hiện tại:
```python
def calculate_interaction_energy_at_r(r_value):
    """Perform the full CP workflow for one R value."""
    energies, statuses = {}, {}
    tasks = {
        "AB": lambda: calculate_dimer_energy(r_value),
        "A_CP": lambda: calculate_monomer_cp_energy(r_value, "MONOMER A"),
        "B_CP": lambda: calculate_monomer_cp_energy(r_value, "MONOMER B"),
    }
    for fragment, task in tasks.items():
        try:
            energies[fragment] = task()
            statuses[fragment] = "OK"
        except Exception as exc:
            energies[fragment] = np.nan
            statuses[fragment] = "Not convergence" if any(token in str(exc).lower() for token in ('not converg', 'unstable')) else f"FAILED: {exc}"
            print(f"{fragment}, R/R0={r_value}: {statuses[fragment]}")
    if not all(status == "OK" for status in statuses.values()):
        return {
            "R_scale": r_value,
            "E_AB_Embed_Eh": energies["AB"],
            "E_A_CP_Embed_Eh": energies["A_CP"],
            "E_B_CP_Embed_Eh": energies["B_CP"],
            "E_int_Eh": np.nan,
            "E_int_kcal_mol": np.nan,
            "Status": "Not convergence" if "Not convergence" in statuses.values() else "FAILED",
            **{f"Status_{fragment}": status for fragment, status in statuses.items()},
        }
    e_ab, e_a_cp, e_b_cp = (energies[k] for k in ("AB", "A_CP", "B_CP"))
```
**Tại sao sửa:** trước đây AB lỗi sẽ bỏ qua A_CP và B_CP tại điểm R đó. Nay từng fragment có try/except riêng, nên vẫn giữ năng lượng của các fragment thành công để chẩn đoán.

Nếu một fragment lỗi, năng lượng fragment đó là `np.nan`, E_int cũng là `np.nan`; CSV/XLSX ghi ô trống và có `Status_AB`, `Status_A_CP`, `Status_B_CP`. Không thay bằng 0. Khi tất cả thành công, phép trừ vẫn là:
```python
    e_int = e_ab - e_a_cp - e_b_cp
    e_int_kcal = e_int * HARTREE_TO_KCAL
```
Không thay đổi định nghĩa counterpoise: cả hai monomer vẫn dùng ghost basis của fragment còn lại.

## 7. Checkpoint kết quả ngay sau mỗi điểm R

**Vị trí:** main loop, mới dòng 417–418; thêm các cột trạng thái mới dòng 426–428.

```python
        # Checkpoint completed points even if a later point is interrupted.
        pd.DataFrame(results_list).to_csv(CSV_FILE, index=False, encoding='utf-8')
```
**Tại sao sửa:** bản cũ chủ yếu ghi bảng sau khi kết thúc scan. Bản mới ghi CSV mỗi điểm, giảm mất kết quả của các điểm đã hoàn thành khi job bị ngắt ở điểm sau. Excel vẫn được xuất cuối scan. Đây chưa phải cơ chế resume tự bỏ qua các điểm đã chạy; dùng lại cùng thư mục có thể ghi đè file.

## 8. Nhãn log chứa R để không ghi đè giữa các hình học

**Vị trí:** `calculate_dimer_energy()` mới dòng 279; `calculate_monomer_cp_energy()` mới dòng 319.

```python
    e_ab = run_embedding_calculation(mol, active_dimer, xc_env='b3lyp', label=f"DIMER_R{r_value:.8g}")
```
```python
    energy = run_embedding_calculation(mol, active_atoms_cp, xc_env='b3lyp', label=f"{label}_CP_R{r_value:.8g}")
```
**Tại sao sửa:** nhãn DIMER hoặc MONOMER A_CP cố định không phân biệt các điểm R; nay tên log chứa R. Giới hạn: `:.8g` giữ 8 chữ số có nghĩa, hai R cực gần nhau hoặc điểm trùng nhau trong cùng run vẫn có thể trùng tên log.

## 9. Dùng một intE cho MP2 và OBMP2; import từ shared

**Vị trí:** cấu hình mới dòng 32–39 và import mới dòng 71–79; khởi tạo mới dòng 189.

```python
REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_DIR = REPO_ROOT / 'shared'
sys.path.insert(0, str(SHARED_DIR))
INPUT_FILE = Path(os.environ.get('INPUT_FILE', str(Path(__file__).resolve().parents[2] / 'inputs/aad.json')))
CONFIG = json.loads(INPUT_FILE.read_text())
METHOD = os.environ.get('METHOD', 'obmp2').lower()
if METHOD not in ('mp2', 'obmp2'):
    raise ValueError('METHOD must be mp2 or obmp2')
```
```python
PYCMF_PATH = os.environ.get("PYCMF_PATH", "/share/installs/pyCMF")
if PYCMF_PATH not in sys.path:
    sys.path.insert(0, PYCMF_PATH)

from pycmf.OBDH.stability import stabilize_scf
if METHOD == 'mp2':
    from mp2_embed import UMP2_CL as EmbeddingMethod
else:
    from pycmf.OBDH import OBMP2_CL as EmbeddingMethod
```
**Tại sao sửa:** đáp ứng yêu cầu tổ chức workspace: `shared/mp2_embed.py` là bản dùng chung; benchmark không giữ một bản riêng. `parents[4]` đi từ file `benchmarks/dimmer_cl/methods/embedding/intE.py` lên gốc repo, không phụ thuộc thư mục đang chạy lệnh. `METHOD=mp2` chọn UMP2_CL; mặc định `obmp2` chọn OBMP2_CL của pyCMF. Đối tượng được tạo bằng `EmbeddingMethod(mf)`.

`PYCMF_PATH` trở thành biến môi trường, mặc định vẫn `/share/installs/pyCMF`; đổi append sang insert(0) để ưu tiên đường dẫn chỉ định. Vì pyCMF được insert sau shared, đường dẫn pyCMF đứng trước shared nếu được thêm mới; không nên có module `mp2_embed` trùng tên trong đó.

**MP2 và OBMP2 không có cùng cờ hội tụ:** UMP2 là phép tính một lần, shared UMP2_CL không có cờ `.converged` của solver OBMP2. Driver chỉ kiểm tra cờ OBMP2 khi METHOD=obmp2; MP2 dựa vào kiểm tra embedded UHF của module shared và kiểm tra năng lượng hữu hạn. Vì vậy không ghi một cờ `MP2_converged=True` giả định.

**Thông tin riêng về module shared:** bản chọn là nguyên bản aldehyde, có embedded UHF max_cycle=200, wrapper get_veff tương thích lời gọi keyword/default và raise khi embedded UHF chưa hội tụ. So với các bản hexanol/mipc cũ, việc dùng chung làm thay đổi cách xử lý hội tụ. Đây là thay đổi tổ chức/module được thực hiện ở bước gom shared, không phải chỉnh công thức trong intE hay source pyCMF.

## 10. Tách hình học/active atoms sang input JSON

**Vị trí:** cấu hình mới dòng 35–36, 67–68; active atoms mới dòng 276, 316.

```python
MONOMER_A_RAW = CONFIG['MONOMER_A_RAW']
MONOMER_B_RAW = CONFIG['MONOMER_B_RAW']
```
```python
    active_dimer = CONFIG['active_atoms']
```
```python
    active_atoms_cp = CONFIG['active_atoms']
```
**Tại sao sửa:** chuyển tọa độ hard-code sang `benchmarks/dimmer_cl/inputs/aad.json`, lưu cả R_VALUES và active_atoms. Đây là dữ liệu dùng chung cho hai method, dễ kiểm tra và thay đổi có chủ đích. Tọa độ gốc và danh sách `[0,1,2,3,8,9,10,11]` được giữ nguyên.

Cùng chỉ số active atom trong AB và CP không đảm bảo cùng số orbital sau SPADE/CL; danh sách ở monomer CP bao gồm cả ghost AO centers. Cách dịch monomer B theo tâm hình học và tạo ghost atoms không bị đổi.

## 11. Tách thư mục của mỗi lần chạy, nối scratch với PySCF

**Vị trí:** mới dòng 40–42, 58–62, hàm `configure_scratch_directory()` dòng 84–91; main dòng 391.

```python
RUN_ID = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_DIR = Path(os.environ.get('RUN_DIR', str(REPO_ROOT / 'results/dimmer_cl' / METHOD / RUN_ID)))
RUN_DIR.mkdir(parents=True, exist_ok=True)
```
```python
CSV_FILE = os.environ.get("CSV_FILE", str(RUN_DIR / "interaction_energies.csv"))
XLSX_FILE = os.environ.get("XLSX_FILE", str(RUN_DIR / "interaction_energies.xlsx"))
CONVERGENCE_FILE = os.environ.get("CONVERGENCE_FILE", str(RUN_DIR / "convergence.csv"))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(RUN_DIR / "logs")))
CONVERGENCE_RECORDS = []
```
```python
def configure_scratch_directory():
    """Use PYSCF_SCRATCH for large temporary files when it is defined."""
    scratch_dir = os.environ.get("PYSCF_SCRATCH")
    if scratch_dir:
        os.makedirs(scratch_dir, exist_ok=True)
        lib.param.TMPDIR = scratch_dir
        print(f"PySCF scratch directory: {scratch_dir}")
    return scratch_dir
```
**Tại sao sửa:** mỗi method/job có CSV, XLSX, convergence.csv và logs riêng. Chạy Python trực tiếp mặc định ghi dưới `results/dimmer_cl/<method>/<run_id>` trong repo. File `.sh` hiện tại đặt RUN_DIR dưới `/data/giahuy/Result/OBMP2_in_DFT/dimmer_cl/<method>/<Slurm-job-id>/`, theo format HPC cũ. Nếu có `PYSCF_SCRATCH`, driver tạo thư mục và đặt `lib.param.TMPDIR` để file tạm PySCF vào scratch.

BASIS mặc định def2-QZVPPD và MEMORY mặc định 90000 MB vẫn là giá trị của driver gốc. Sửa `.sh` là thay đổi launcher riêng, không phải sửa công thức tính của intE.

## 12. Các phần được giữ nguyên và bằng chứng kiểm tra

Giữ nguyên hình học, cách scale R/R0, ghost basis/counterpoise, B3LYP, active atom indices mặc định, mu=1e6, CL hai shells, second_order=True, frozen=None, thresh=1e-6 và hệ số 627.509474 kcal/mol/Eh. Không sửa source pyCMF trong bản bàn giao này. Chú thích đầu file được cập nhật để mô tả đúng luồng thực thi; việc đổi mô tả OBMP2 thành One-Body MP2 ở bản driver trung gian không phải thay đổi thuật toán.

Các thay đổi có tác động đến cách chạy/nhận kết quả gồm UHF tolerance/max_cycle, bỏ ép incore, số vòng OBMP2/DIIS, gọi kernel một lần, kiểm tra hội tụ/stability và loại bỏ kết quả lỗi. Vì vậy không nên mô tả toàn bộ sửa đổi là “chỉ đổi đường dẫn”.

### Kết quả đã test với basis STO-3G

Driver trước khi chuyển vào repo đã chạy cả ba fragment ở 5 điểm, với backend pyCMF không sửa:

| R/R0 | OBMP2-in-B3LYP E_int (kcal/mol) | MP2-in-B3LYP E_int (kcal/mol) |
| --- | ---: | ---: |
| 0.95 | 2.294246576 | 4.306001511 |
| 1.00 | -4.374280788 | -3.253033061 |
| 1.05 | -4.910620348 | -4.303478821 |
| 1.25 | -1.773104924 | -1.667589393 |
| 2.00 | -0.091386035 | -0.085177145 |

Bản `intE.py` trong repo được chạy lại full CP tại R/R0=1.0: MP2 = -3.253033029 kcal/mol; OBMP2 = -4.374280755 kcal/mol. Cả AB, A_CP, B_CP có Status OK; sai khác với driver trước nhỏ hơn 1e-8 Eh. Đã kiểm tra thêm chuẩn hóa R đơn/list, một lần gọi kernel, CP subtraction, nhánh fragment lỗi/NaN, và từ chối SCF failure hiện rõ trong log.

**Giới hạn kết luận:** test basis nhỏ xác nhận pipeline thực thi và xu hướng năng lượng trong cấu hình đó. Không xác nhận hội tụ def2-QZVPPD hay nguyên nhân anomaly ở basis lớn; không đủ để so độ chính xác với CCSD(T). Ở STO-3G, CL giữ hết effective virtual trong các phép thử này, nên chưa kiểm chứng sai số do virtual truncation mạnh.

## Phụ lục: diff đầy đủ file gốc → intE.py hiện tại

Diff dưới đây chỉ phục vụ đối chiếu. File gốc và file hiện tại có đường dẫn/tên khác nhau; không dùng diff này như một patch áp mù lên file khác.

```diff
--- intE_obmp2inb3lyp.py (original)
+++ benchmarks/dimmer_cl/methods/embedding/intE.py (current)
@@ -1,11 +1,12 @@
 """
-AAD Interaction Energy: OBMP2-in-B3LYP (Embedding + CL Truncation) R-scan.
-
-This script executes an Orbital-Optimized MP2 (OBMP2) in DFT (B3LYP) embedding 
+AAD interaction energy: MP2 or OBMP2 in B3LYP, with SPADE, CL and counterpoise.
+
+This script executes the selected MP2 or OBMP2 in DFT (B3LYP) embedding
 R-scan calculation with Counterpoise (CP) correction.
 
 Workflow:
-    UHF -> SCF Stabilization -> B3LYP-in-UHF Embedding -> OBMP2-in-B3LYP (SPADE + CL)
+    UHF reference + stability -> full-system B3LYP -> SPADE -> embedded UHF
+    -> CL -> OBMP2 -> recombined embedding total energy
 for the dimer and for each ghost-atom-corrected monomer (Counterpoise).
 
 Interaction energy (CP corrected):
@@ -17,6 +18,9 @@
 import sys
 import time
 import json
+from datetime import datetime
+import contextlib
+from pathlib import Path
 import numpy as np
 import pandas as pd
 from pyscf import lib, scf, gto, ao2mo
@@ -25,61 +29,54 @@
 # 1. CONFIGURATION & GEOMETRY
 # ==============================================================================
 
+REPO_ROOT = Path(__file__).resolve().parents[4]
+SHARED_DIR = REPO_ROOT / 'shared'
+sys.path.insert(0, str(SHARED_DIR))
+INPUT_FILE = Path(os.environ.get('INPUT_FILE', str(Path(__file__).resolve().parents[2] / 'inputs/aad.json')))
+CONFIG = json.loads(INPUT_FILE.read_text())
+METHOD = os.environ.get('METHOD', 'obmp2').lower()
+if METHOD not in ('mp2', 'obmp2'):
+    raise ValueError('METHOD must be mp2 or obmp2')
+RUN_ID = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
+RUN_DIR = Path(os.environ.get('RUN_DIR', str(REPO_ROOT / 'results/dimmer_cl' / METHOD / RUN_ID)))
+RUN_DIR.mkdir(parents=True, exist_ok=True)
+
 # Nhận tham số từ môi trường Slurm/Bash
 ENV_R = os.environ.get("R")
 # Mặc định chuỗi scan R nếu không truyền từ môi trường
-R_VALUES = json.loads(ENV_R) if ENV_R else [0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50, 2.00]
+R_VALUES = json.loads(ENV_R) if ENV_R else CONFIG['R_VALUES']
+if isinstance(R_VALUES, (int, float)):
+    R_VALUES = [R_VALUES]
+if not isinstance(R_VALUES, list) or not R_VALUES:
+    raise ValueError("R must be a number or nonempty JSON list")
+R_VALUES = [float(r) for r in R_VALUES]
+if any(not np.isfinite(r) or r <= 0 for r in R_VALUES):
+    raise ValueError("R values must be finite and positive")
 
 BASIS = os.environ.get("BASIS", "def2-QZVPPD")
 MEMORY = int(os.environ.get("MEMORY", 90000)) # MB
-CSV_FILE = os.environ.get("CSV_FILE", "intE_obmp2inb3lyp.csv")
-XLSX_FILE = os.environ.get("XLSX_FILE", "intE_obmp2inb3lyp.xlsx")
+CSV_FILE = os.environ.get("CSV_FILE", str(RUN_DIR / "interaction_energies.csv"))
+XLSX_FILE = os.environ.get("XLSX_FILE", str(RUN_DIR / "interaction_energies.xlsx"))
+CONVERGENCE_FILE = os.environ.get("CONVERGENCE_FILE", str(RUN_DIR / "convergence.csv"))
+LOG_DIR = Path(os.environ.get("LOG_DIR", str(RUN_DIR / "logs")))
+CONVERGENCE_RECORDS = []
 
 # Hằng số vật lý
 HARTREE_TO_KCAL = 627.509474
 
-# Tọa độ gốc Monomer A (Acetic Acid 1 - 8 atoms)
-MONOMER_A_RAW = [
-    ["C", -1.061709204,  1.297140572,  0.292060003],
-    ["O", -0.358161116,  2.270458613,  0.531812668],
-    ["O", -0.589303516,  0.094917758,  0.003788813],
-    ["H",  0.404435659,  0.127722621,  0.018411838],
-    ["C", -2.558427798,  1.342549823,  0.296257320],
-    ["H", -2.895997978,  2.347464002,  0.518316340],
-    ["H", -2.932889278,  1.022390451, -0.672995551],
-    ["H", -2.937211960,  0.644910433,  1.039557084]
-]
-
-# Tọa độ gốc Monomer B (Acetic Acid 2 - 8 atoms)
-MONOMER_B_RAW = [
-    ["C",  2.799564974,  1.108464452,  0.271261944],
-    ["O",  2.095946609,  0.135149964,  0.031474342],
-    ["O",  2.327138640,  2.310899840,  0.559040415],
-    ["H",  1.333350100,  2.278001606,  0.544639911],
-    ["C",  4.296277422,  1.062561706,  0.269297549],
-    ["H",  4.633856986,  0.061242513,  0.031772054],
-    ["H",  4.677775967,  1.772914645, -0.460171349],
-    ["H",  4.667793733,  1.365256223,  1.245352910]
-]
+MONOMER_A_RAW = CONFIG['MONOMER_A_RAW']
+MONOMER_B_RAW = CONFIG['MONOMER_B_RAW']
 
 # Thêm đường dẫn pyCMF
-PYCMF_PATH = "/share/installs/pyCMF"
+PYCMF_PATH = os.environ.get("PYCMF_PATH", "/share/installs/pyCMF")
 if PYCMF_PATH not in sys.path:
-    sys.path.append(PYCMF_PATH)
-
-# Import module OBMP2 Embedding
-try:
-    from pycmf.OBDH import OBMP2_CL
-    from pycmf.OBDH.stability import stabilize_scf
-except ImportError:
-    try:
-        from OBDH import OBMP2_CL
-        from OBDH.stability import stabilize_scf
-    except ImportError:
-        print("[ERROR] Could not import OBMP2_CL and stabilize_scf from pyCMF.")
-        print("Make sure pyCMF is properly installed in PYCMF_PATH.")
-        sys.exit(1)
-
+    sys.path.insert(0, PYCMF_PATH)
+
+from pycmf.OBDH.stability import stabilize_scf
+if METHOD == 'mp2':
+    from mp2_embed import UMP2_CL as EmbeddingMethod
+else:
+    from pycmf.OBDH import OBMP2_CL as EmbeddingMethod
 
 # ==============================================================================
 # 2. UTILITIES & HELPER FUNCTIONS
@@ -112,38 +109,6 @@
     mol.build()
     return mol
 
-def _extract_total_energy(result, calculation):
-    """Safely obtain the total OBMP2 energy from calculation object or result."""
-    if isinstance(result, (float, int, np.floating)):
-        return float(result)
-
-    e_corr = getattr(calculation, "e_corr", None)
-    scf_obj = getattr(calculation, "_scf", None)
-    scf_e_tot = getattr(scf_obj, "e_tot", None)
-
-    if e_corr is not None and scf_e_tot is not None:
-        return float(e_corr + scf_e_tot)
-
-    if result is not None and result is not calculation:
-        result_corr = getattr(result, "e_corr", None)
-        result_scf = getattr(getattr(result, "_scf", None), "e_tot", None)
-        if result_corr is not None and result_scf is not None:
-            return float(result_corr + result_scf)
-        if hasattr(result, "e_tot") and getattr(result, "e_tot") is not None:
-            return float(result.e_tot)
-
-    if hasattr(calculation, "e_tot") and getattr(calculation, "e_tot") is not None:
-        try:
-            return float(calculation.e_tot)
-        except TypeError:
-            pass
-
-    raise RuntimeError(
-        f"{calculation.__class__.__name__} finished without a valid "
-        f"OBMP2 correlation energy: e_corr={e_corr!r}, SCF energy={scf_e_tot!r}."
-    )
-
-
 # ==============================================================================
 # 3. GEOMETRY GENERATION
 # ==============================================================================
@@ -154,65 +119,74 @@
     """
     coords_A = np.array([[x, y, z] for _, x, y, z in MONOMER_A_RAW])
     coords_B = np.array([[x, y, z] for _, x, y, z in MONOMER_B_RAW])
-    
+
     center_A = np.mean(coords_A, axis=0)
     center_B = np.mean(coords_B, axis=0)
-    
+
     R_vec = center_B - center_A
     shift_vec = (scale_factor - 1.0) * R_vec
-    
+
     scaled_B_raw = []
     for i, (atom, x, y, z) in enumerate(MONOMER_B_RAW):
         new_xyz = coords_B[i] + shift_vec
         scaled_B_raw.append([atom, new_xyz[0], new_xyz[1], new_xyz[2]])
-        
+
     str_A_real = format_coords(MONOMER_A_RAW, is_ghost=False)
     str_A_ghost = format_coords(MONOMER_A_RAW, is_ghost=True)
     str_B_real = format_coords(scaled_B_raw, is_ghost=False)
     str_B_ghost = format_coords(scaled_B_raw, is_ghost=True)
-    
+
     dimer_geo = str_A_real + "\n" + str_B_real
     monA_cp_geo = str_A_real + "\n" + str_B_ghost
     monB_cp_geo = str_A_ghost + "\n" + str_B_real
-    
+
     return dimer_geo, monA_cp_geo, monB_cp_geo
 
 
 # ==============================================================================
 # 4. CORE CALCULATION DRIVER (OBMP2-IN-B3LYP)
 # ==============================================================================
-def run_embedding_calculation(mol, active_atoms, xc_env='b3lyp', label=""):
-    """
-    Chạy quy trình: UHF -> Stability Check -> B3LYP-in-UHF -> OBMP2-in-B3LYP (SPADE+CL)
+def _run_embedding_calculation(mol, active_atoms, xc_env='b3lyp', label="", record=None):
+    """
+    UHF reference/stability, then backend full B3LYP -> SPADE -> embedded UHF -> CL -> OBMP2.
     """
     print(f"\n--- UHF Reference for {label} ---")
-    mol.incore_anyway = True
-    
+    mol.incore_anyway = False
+
     mf = scf.UHF(mol).density_fit()
     mf.max_memory = MEMORY
     mf.verbose = mol.verbose
-    
+    mf.conv_tol = 1e-10
+    mf.max_cycle = 200
+
     start_scf = time.time()
     mf.kernel()
+    record['UHF_converged'] = bool(mf.converged)
     if not mf.converged:
-        print(f"[WARNING] UHF did not converge for {label}.")
+        raise RuntimeError(f"{label}: UHF Not convergence")
     print(f"UHF runtime = {time.time() - start_scf:.3f} s")
     print(f"UHF Energy = {mf.e_tot:.12f} Eh")
 
     # Kiểm tra và ổn định nghiệm SCF
     mf = stabilize_scf(mf, max_macro_cycles=10, verbose=True)
+    if not mf.converged:
+        raise RuntimeError(f"{label}: stabilized UHF Not convergence")
+    _, _, stable, _ = mf.stability(return_status=True)
+    record['UHF_stable'] = bool(stable)
+    if not stable:
+        raise RuntimeError(f"{label}: unstable UHF after stabilization")
 
     # Nén ERI nếu có
     if hasattr(mf, "_eri") and mf._eri is not None:
         if mf._eri.ndim == 4:
             mf._eri = ao2mo.restore(8, mf._eri, mol.nao)
-            
-    gc.collect()
-
-    print(f"\n--- OBMP2-in-B3LYP Embedding for {label} ---")
-    
+
+    gc.collect()
+
+    print(f"\n--- {METHOD.upper()}-in-B3LYP Embedding for {label} ---")
+
     # Khởi tạo đối tượng OBMP2_CL
-    mp_emb = OBMP2_CL(mf)
+    mp_emb = EmbeddingMethod(mf)
     mp_emb.thresh = 1e-6
     mp_emb.second_order = True
     mp_emb.use_embed = True
@@ -225,51 +199,69 @@
     mp_emb.is_hybrid = False
     mp_emb.with_df = mf.with_df
     mp_emb.max_memory = MEMORY
-    if not hasattr(mp_emb, "niter"):
-        mp_emb.niter = 1000
-    
+    mp_emb.niter = int(os.environ.get("OBMP2_NITER", "1000"))
+    mp_emb.diis_space = int(os.environ.get("DIIS_SPACE", "8"))
+
     start_embed = time.time()
-    result = None
-    
-    # Chạy tính toán OBMP2 (thử .run(), nếu không có/lỗi thì chuyển qua .kernel())
-    if hasattr(mp_emb, "run"):
-        try:
-            result = mp_emb.run()
-        except Exception as e:
-            print(f"[WARNING] mp_emb.run() gặp sự cố: {e}")
-
-    if getattr(mp_emb, "e_corr", None) is None and hasattr(mp_emb, "kernel"):
-        print(f"[INFO] Running OBMP2 kernel for {label}...")
-        result = mp_emb.kernel()
-
-    # Bổ sung e_corr dự phòng nếu result trả về giá trị
-    if getattr(mp_emb, "e_corr", None) is None and result is not None:
-        if isinstance(result, (float, int, np.floating)):
-            scf_e = getattr(getattr(mp_emb, "_scf", None), "e_tot", 0.0)
-            mp_emb.e_corr = float(result) - scf_e
-        elif isinstance(result, (tuple, list)) and len(result) > 0:
-            if isinstance(result[0], (float, int, np.floating)):
-                mp_emb.e_corr = float(result[0])
-
+    e_tot = float(mp_emb.kernel())
+    if METHOD == 'obmp2':
+        record['OBMP2_converged'] = bool(getattr(mp_emb, 'converged', False))
+        record['active_nocc'] = str(tuple(int(n) for n in mp_emb.get_nocc()))
+        record['active_nmo'] = str(tuple(int(n) for n in mp_emb.get_nmo()))
+    else:
+        record['Solver'] = 'one-shot UMP2; embedded UHF convergence checked by shared module'
+    if not np.isfinite(e_tot):
+        raise RuntimeError(f"{label}: non-finite embedding total energy")
+    if METHOD == "obmp2" and not bool(getattr(mp_emb, "converged", False)):
+        raise RuntimeError(f"{label}: OBMP2 Not convergence")
     print(f"Embedding runtime = {time.time() - start_embed:.3f} s")
-    
-    e_tot = _extract_total_energy(result, mp_emb)
-    
+
     if hasattr(mf, "_eri"): del mf._eri
     del mp_emb, mf
     gc.collect()
-    
+
     return e_tot
 
 
+def run_embedding_calculation(mol, active_atoms, xc_env='b3lyp', label=""):
+    """Call the unchanged backend once; retain fragment diagnostics on failure."""
+    LOG_DIR.mkdir(parents=True, exist_ok=True)
+    log_path = LOG_DIR / f"{label.replace(' ', '_')}.log"
+    record = dict(Method=METHOD, Label=label, Basis=BASIS, Status="FAILED", Log=str(log_path))
+    original_stdout = mol.stdout
+    print(f"Running {label}; log: {log_path}")
+    try:
+        with log_path.open('w', buffering=1) as log:
+            mol.stdout = log
+            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
+                energy = _run_embedding_calculation(mol, active_atoms, xc_env, label, record)
+        # The backend does not expose the DFT and embedded-UHF SCF objects.
+        # Reject an explicit SCF failure reported in its calculation log.
+        backend_log = log_path.read_text().split(f'--- {METHOD.upper()}-in-B3LYP Embedding', 1)[-1]
+        backend_scf_failed = 'SCF not converged' in backend_log
+        record['Backend_SCF_no_failure_reported'] = not backend_scf_failed
+        if backend_scf_failed:
+            raise RuntimeError(f"{label}: backend SCF Not convergence; see {log_path}")
+        record.update(Status="OK", E_total_Eh=energy)
+        return energy
+    except Exception as exc:
+        record['Status'] = "Not convergence" if any(token in str(exc).lower() for token in ('not converg', 'unstable')) else "FAILED"
+        record['Error'] = str(exc)
+        raise
+    finally:
+        mol.stdout = original_stdout
+        CONVERGENCE_RECORDS.append(record)
+        pd.DataFrame(CONVERGENCE_RECORDS).to_csv(CONVERGENCE_FILE, index=False)
+
+
 # ==============================================================================
 # 5. INDIVIDUAL ENERGY ROUTINES
 # ==============================================================================
 def calculate_dimer_energy(r_value):
-    """Calculate OBMP2-in-B3LYP total energy of the AAD dimer at one R."""
-    gc.collect() 
+    """Calculate the selected method in B3LYP total energy of the AAD dimer at one R."""
+    gc.collect()
     print("\n" + "#" * 80)
-    print(f"#       AAD OBMP2-in-B3LYP EMBED: R_scale = {r_value:.2f}")
+    print(f"#       AAD {METHOD.upper()}-in-B3LYP EMBED: R_scale = {r_value:.2f}")
     print("#" * 80)
 
     dimer_geo, _, _ = get_shifted_geometries(r_value)
@@ -279,17 +271,17 @@
     print("------------------")
     print(f"R_scale             : {r_value:.2f}")
     print(f"Basis               : {BASIS}")
-    
+
     # Active region Dimer: nhóm -COOH của MonA (0-3) và MonB (8-11)
-    active_dimer = [0, 1, 2, 3, 8, 9, 10, 11]
-    
+    active_dimer = CONFIG['active_atoms']
+
     start_time = time.time()
-    e_ab = run_embedding_calculation(mol, active_dimer, xc_env='b3lyp', label="DIMER")
+    e_ab = run_embedding_calculation(mol, active_dimer, xc_env='b3lyp', label=f"DIMER_R{r_value:.8g}")
     runtime = time.time() - start_time
 
     print("\nDIMER RESULT")
     print("------------")
-    print(f"E_AB (OBMP2-in-B3LYP) = {e_ab:.12f} Eh")
+    print(f"E_AB ({METHOD.upper()}-in-B3LYP) = {e_ab:.12f} Eh")
     print(f"Total Runtime = {runtime:.3f} s")
 
     del mol
@@ -299,17 +291,17 @@
 
 def calculate_monomer_cp_energy(r_value, label):
     """
-    Calculate the OBMP2-in-B3LYP energy of one monomer with Ghost atoms (Counterpoise).
-    Giữ nguyên Active Space [0, 1, 2, 3, 8, 9, 10, 11] trùng khớp hoàn toàn với Dimer.
+    Calculate the {METHOD.upper()}-in-B3LYP energy of one monomer with Ghost atoms (Counterpoise).
+    Giữ nguyên Active Space CONFIG['active_atoms'] trùng khớp hoàn toàn với Dimer.
     """
     gc.collect()
     print("\n" + "#" * 80)
-    print(f"#       {label} (Counterpoise): OBMP2-in-B3LYP EMBED")
+    print(f"#       {label} (Counterpoise): {METHOD.upper()}-in-B3LYP EMBED")
     print(f"#       R_scale = {r_value:.2f}")
     print("#" * 80)
 
     _, monA_cp_geo, monB_cp_geo = get_shifted_geometries(r_value)
-    
+
     if label == "MONOMER A":
         geo = monA_cp_geo
     else:
@@ -320,11 +312,11 @@
     print(f"\n{label} information")
     print("-------------------")
     print(f"Basis               : {BASIS}")
-    
-    active_atoms_cp = [0, 1, 2, 3, 8, 9, 10, 11]
+
+    active_atoms_cp = CONFIG['active_atoms']
 
     start_time = time.time()
-    energy = run_embedding_calculation(mol, active_atoms_cp, xc_env='b3lyp', label=f"{label}_CP")
+    energy = run_embedding_calculation(mol, active_atoms_cp, xc_env='b3lyp', label=f"{label}_CP_R{r_value:.8g}")
     runtime = time.time() - start_time
 
     print(f"\n{label} CP Energy = {energy:.12f} Eh")
@@ -336,13 +328,36 @@
 
 
 # ==============================================================================
-# 6. INTERACTION ENERGY WORKFLOW 
+# 6. INTERACTION ENERGY WORKFLOW
 # ==============================================================================
 def calculate_interaction_energy_at_r(r_value):
     """Perform the full CP workflow for one R value."""
-    e_ab = calculate_dimer_energy(r_value)
-    e_a_cp = calculate_monomer_cp_energy(r_value, "MONOMER A")
-    e_b_cp = calculate_monomer_cp_energy(r_value, "MONOMER B")
+    energies, statuses = {}, {}
+    tasks = {
+        "AB": lambda: calculate_dimer_energy(r_value),
+        "A_CP": lambda: calculate_monomer_cp_energy(r_value, "MONOMER A"),
+        "B_CP": lambda: calculate_monomer_cp_energy(r_value, "MONOMER B"),
+    }
+    for fragment, task in tasks.items():
+        try:
+            energies[fragment] = task()
+            statuses[fragment] = "OK"
+        except Exception as exc:
+            energies[fragment] = np.nan
+            statuses[fragment] = "Not convergence" if any(token in str(exc).lower() for token in ('not converg', 'unstable')) else f"FAILED: {exc}"
+            print(f"{fragment}, R/R0={r_value}: {statuses[fragment]}")
+    if not all(status == "OK" for status in statuses.values()):
+        return {
+            "R_scale": r_value,
+            "E_AB_Embed_Eh": energies["AB"],
+            "E_A_CP_Embed_Eh": energies["A_CP"],
+            "E_B_CP_Embed_Eh": energies["B_CP"],
+            "E_int_Eh": np.nan,
+            "E_int_kcal_mol": np.nan,
+            "Status": "Not convergence" if "Not convergence" in statuses.values() else "FAILED",
+            **{f"Status_{fragment}": status for fragment, status in statuses.items()},
+        }
+    e_ab, e_a_cp, e_b_cp = (energies[k] for k in ("AB", "A_CP", "B_CP"))
 
     e_int = e_ab - e_a_cp - e_b_cp
     e_int_kcal = e_int * HARTREE_TO_KCAL
@@ -365,6 +380,7 @@
         "E_int_Eh": e_int,
         "E_int_kcal_mol": e_int_kcal,
         "Status": "OK",
+        **{f"Status_{fragment}": status for fragment, status in statuses.items()},
     }
 
 
@@ -373,44 +389,46 @@
 # ==============================================================================
 if __name__ == "__main__":
     configure_scratch_directory()
-    
+
     print("\n" + "="*95)
-    print(" >>> STARTING AAD OBMP2-in-B3LYP EMBEDDING R-SCAN (Hệ số nhân) <<<")
+    print(f" >>> STARTING AAD {METHOD.upper()}-in-B3LYP EMBEDDING R-SCAN (Hệ số nhân) <<<")
     print("="*95)
-    
+
     results_list = []
     total_start = time.time()
-    
+
     for idx, r in enumerate(R_VALUES, start=1):
         try:
             point_start = time.time()
             print(f"\n[SCAN POINT {idx}/{len(R_VALUES)}] Chạy cho R_scale = {r:.2f}")
-            
+
             result_point = calculate_interaction_energy_at_r(r)
             result_point["Point"] = idx
             result_point["Runtime_Point_s"] = time.time() - point_start
-            
+
             results_list.append(result_point)
-            
+
         except Exception as e:
             print(f"\n[ERROR] Lỗi tại điểm R = {r}: {e}")
             results_list.append({
                 "Point": idx, "R_scale": r, "Status": f"FAILED: {str(e)}"
             })
-            continue
-            
+
+        # Checkpoint completed points even if a later point is interrupted.
+        pd.DataFrame(results_list).to_csv(CSV_FILE, index=False, encoding='utf-8')
+
     print("\n" + "="*95)
     print(f" >>> HOÀN THÀNH R-SCAN. Tổng thời gian: {time.time() - total_start:.2f} s <<<")
     print("="*95)
-    
+
     df = pd.DataFrame(results_list)
-    
-    cols_order = ["Point", "R_scale", "E_int_kcal_mol", "E_int_Eh", 
-                  "E_AB_Embed_Eh", "E_A_CP_Embed_Eh", "E_B_CP_Embed_Eh", 
-                  "Runtime_Point_s", "Status"]
+
+    cols_order = ["Point", "R_scale", "E_int_kcal_mol", "E_int_Eh",
+                  "E_AB_Embed_Eh", "E_A_CP_Embed_Eh", "E_B_CP_Embed_Eh",
+                  "Runtime_Point_s", "Status", "Status_AB", "Status_A_CP", "Status_B_CP"]
     existing_cols = [c for c in cols_order if c in df.columns]
     df = df[existing_cols]
-    
+
     # 1. Xuất Excel
     try:
         df.to_excel(XLSX_FILE, index=False, engine='openpyxl')
@@ -418,7 +436,7 @@
     except ImportError:
         print("\n[WARNING] Không thể xuất Excel vì thiếu thư viện 'openpyxl'.")
         print("Hãy cài đặt bằng: pip install openpyxl")
-        
+
     # 2. Xuất CSV (backup)
     df.to_csv(CSV_FILE, index=False, encoding='utf-8')
-    print(f"[+] Đã xuất thành công file CSV:   {CSV_FILE}")+    print(f"[+] Đã xuất thành công file CSV:   {CSV_FILE}")

```
