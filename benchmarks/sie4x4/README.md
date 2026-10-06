# SIE4x4: guesses, nghiệm và benchmark

`tools/run_sie.py` là protocol chung. `methods/dft/calc_DFT.py` và
`methods/obdh/calc_OBDH.py` chỉ là entry point; không giữ hai bản logic SCF.
Input hình học duy nhất: `inputs/input.json`. Kết quả cũ trong `results/sie4x4/*/legacy`
được giữ để tham khảo, không coi là kết quả đã xác minh của pipeline mới.

## Điều kiện từ bài báo

Bao, Gagliardi, Truhlar, JPCL 2018, DOI 10.1021/acs.jpclett.8b00242;
đối chiếu bài chính và SI Table S8, Table S4.

| Điều kiện | Pipeline |
|---|---|
| 4 hệ cation, R/Re = 1, 1.25, 1.5, 1.75 | 16 reaction energies, cần 20 supermolecule calculations cho mỗi phương pháp |
| Điện tích +1, doublet | PySCF `charge=1, spin=1` (Nα−Nβ=1); ORCA multiplicity 2 |
| Cho phép phá đối xứng không gian và spin | UHF/UKS, không bật symmetry; không ép ⟨S²⟩=0.75 |
| SCF hội tụ và nghiệm ổn định | Kiểm tra riêng convergence + internal stability; theo orbital bất ổn rồi chạy và kiểm tra lại |
| Chọn nghiệm SCF biến phân thấp nhất tìm được | UHF/DFT; UMP2 và one-shot DH dùng reference SCF thấp nhất, không tối thiểu hóa năng lượng PT2 qua các reference |
| De = E(DL) − E(R) | Cả hai là supermolecule, không cộng năng lượng monomer, không BSSE correction bổ sung |
| H₂⁺ có He spectator trên trục, cách H gần nhất 6 Å | He là nhóm population riêng; không gộp nhầm thành fragment H₂ |
| Khoảng cách DL | H–H = 10Re = 10.5727436 Å; He–He = 10Re = 10.7420374 Å; N–N/O–O = 15 Å |
| Basis/grid của bài | aug-cc-pVTZ/ultrafine; mặc định bản kiểm chứng này aug-cc-pVDZ/PySCF grid level 4 |

Các tọa độ đầu vào đã được đối chiếu với SI, kể cả điểm NH₃ R=1.5 có trục xoay.
PySCF grid level không đồng nhất định nghĩa Gaussian ultrafine hay ORCA DEFGRID3.
Muốn đối chiếu số chính xác cần kiểm tra hội tụ basis/grid/DF; không bắt buộc cho
kiểm chứng pipeline basis nhỏ. Sai số so với W2-F12 trong bảng là sai số reaction
energy tổng, không riêng một phép đo SIE tinh khiết. MUE cationic 16 điểm khác
AverageMUE gồm cả neutral trong Table 3.

## Init guess và multiguess

1. Từng nguồn UHF và UKS được thử `fragment_average`, `minao`, `atom`, `huckel`, localized A và B. Thêm orbitals/occupations của nghiệm ổn định từ nguồn đã chạy trước làm warm guess (UHF trước, matched UKS, PBE0 rồi PBE).
   UKS ở đây là unrestricted KS với cùng hệ số trao đổi/tương quan của OBDH;
   không phải noncollinear GKS.
2. Localized guesses lấy full density block của fragment, giữ cả các phần tử AO
   nối các nguyên tử khác nhau trong cùng NH₃/H₂O. Electron allocation tổng là doublet.
   Hai phía mang điện tích đều được thử; mật độ trung bình A/B là guess deloc với spectator trung hòa. Fractional occupations chỉ ở guess, không dùng smearing/fractional occupation cho nghiệm cuối. Guess chỉ là điểm bắt đầu, không là nhãn nghiệm.
3. SCF không hội tụ được thử level shift tạm thời, sau đó bỏ shift và hội tụ lại, rồi thử Newton nếu cần; nghiệm bất ổn được theo hướng orbital của
   stability analysis, tối đa số vòng cấu hình. Hessian ổn định không thay thế hội tụ. Kiểm tra cả dE và orbital gradient (mặc định 1e-9 Eh và 1e-6); recompute gradient cuối thay vì chỉ tin cờ converged.
4. Warm guesses chạy Newton trực tiếp từ orbitals/occupations; việc chỉ đọc lại density rồi diagonalize Aufbau có thể làm mất basin và đưa lỗ điện tích lên He. Theo hướng instability cũng dùng Newton với orbital rotation trả về. Không dừng sớm vì hai energy gần nhau. Chỉ deduplicate density đã hội tụ/ổn định
   trong cùng nguồn, bằng Frobenius norm ở AO trực giao (mặc định 1e-5).
5. Với OBDH/OBMP2, chạy tiếp **mọi** mật độ SCF khác nhau hợp lệ từ cả hai nguồn.
   Orbital UKS được đưa vào UHF carrier **không chạy lại UHF SCF**, để giữ basin
   nhưng dùng HF operator đúng trong BCH/subtraction. UKS object không được làm HF operator.
6. Giữ điều kiện solver OB hiện có: dE ≤ 1e-6 Eh và norm effective occupied–virtual
   Fock block ≤ 1e-6, tối đa 300 vòng mặc định. Ghi cycle/dE/Fia/dRMS cuối.
   Fia này không phải chứng minh đạo hàm chính xác dE/dκ hay Hessian OBDH.
7. Với OB, chọn energy thấp nhất trong các nghiệm hội tụ đã tìm được. Đây là quy
   tắc báo cáo thực dụng cho phương pháp đang triển khai, không chứng minh minimum
   toàn cục hay variational stability của OBDH. Stability SCF của seed không chứng nhận OB cuối.

Bảng `init_comparison` ghi HF-init và UKS-init riêng. Bảng `candidates` giữ tất cả
SCF/OB candidates, lỗi, trạng thái convergence và spin. Không dùng fallback từ
nghiệm OB chưa hội tụ. Nhãn cuối tính từ orbital density `ob.gamma`, không từ seed.

Spin populations và charges là Mulliken **tổng theo fragment**. Nhãn mô tả dùng
f = |sA−sB|/(|sA|+|sB|): f≥0.8 → loc; f≤0.2 → deloc; còn lại mixed.
Đây là quy ước chẩn đoán phụ thuộc basis/population, không là tiêu chí loại nghiệm.
⟨S²⟩ được tính cho determinant từ orbitals cuối, không là correlated spin observable.
Luôn đọc các population liên tục cùng energy thay vì chỉ đọc nhãn loc/deloc.

Mặc định α=(0.53,0.39) được **giữ nguyên** từ benchmark cũ. `dh_matched` dùng
E_KS + 0.39 E_MP2 trên KS orbitals, làm đối chứng cùng hệ số. Không gọi bộ này
B2PLYP chuẩn. B2PLYP có (0.53,0.27): có thể chạy một outdir riêng với
`--alpha 0.53 0.27`. Đây là biến thể nghiên cứu bổ sung, không tự ý đổi định nghĩa OBDH.

## Cài và chạy

Bản vá pyCMF dựa trên upstream `af3614dcd89b78f68b2d5912ddd33a2d277f631a`.
Không ghi được trực tiếp lên repo pyCMF từ kết nối hiện tại; diff và installer
được lưu trong `patches/` và `tools/patch_pycmf.py`. Installer kiểm tra hash,
dry-run trước, giữ `.pre-sie-fix`, và không áp dụng mù lên phiên bản khác.

Từ root `job_HPC`, với environment có PySCF, NumPy, SciPy, pandas, openpyxl, pyCMF:

```bash
python benchmarks/sie4x4/tools/patch_pycmf.py
python -m unittest discover -s benchmarks/sie4x4/tests -v
python benchmarks/sie4x4/tools/run_sie.py \
  --outdir results/sie4x4/validation/avdz-grid4 \
  --basis aug-cc-pVDZ --grid 4 --threads 4
```

Nếu dùng checkout pyCMF editable: `python benchmarks/sie4x4/tools/patch_pycmf.py --root /path/to/pyCMF`.
Nếu phiên bản cài đặt khác: cài checkout đúng commit trên trong environment nghiên cứu
riêng rồi áp dụng patch; script sẽ báo lỗi thay vì sửa phiên bản không biết.

Thử nhanh trước khi full benchmark:

```bash
python benchmarks/sie4x4/tools/run_sie.py \
  --outdir results/sie4x4/validation/quick \
  --basis aug-cc-pVDZ --grid 3 --threads 2 \
  --systems He2_plus --points R_1.0 dissociation_limit
```

SLURM: submit từ root repo; activate environment trước, hoặc đặt `VENV`.
Không còn đường dẫn tuyệt đối phụ thuộc `/home/giahuy/Code/job/OBDH/...` cũ.

```bash
sbatch benchmarks/sie4x4/jobs/dft/job.sh
sbatch benchmarks/sie4x4/jobs/obdh/job.sh
```

Có thể đặt `BASIS`, `GRID`, `OUTDIR`, `REPO_ROOT`, `JOB_SCRATCH_PATH` qua environment.
`--methods`, `--systems`, `--points`, `--alpha` và các tolerance được lưu trong config.
Thread count theo `SLURM_CPUS_PER_TASK`. Bản DF mặc định dùng
`def2-universal-jkfit` cho SCF và auxiliary MP2-fit tự chọn riêng cho OB/UMP2.

Từng case có `.log` và `.json` checkpoint, rồi tổng hợp CSV + `SIE4x4.xlsx`.
Resume chỉ khi config, input, driver và pyCMF hash khớp. Đổi basis/alpha/grid/code
phải dùng outdir mới. Đường dẫn point giữ nguyên `R_1.0`, `R_1.25`, ... không bị
mất phần thập phân. Job exit nonzero khi có selected energy thiếu.
`MUE_16` chỉ hiện nếu đủ cả 16 De hợp lệ; `MUE_available` kèm n_valid chỉ mô tả
subset; `common_statistics` dùng giao điểm hợp lệ giữa các method được yêu cầu.

## ORCA là đối chứng tùy chọn

```bash
export ORCA_DIR=/path/to/orca
sbatch benchmarks/sie4x4/jobs/orca/job_SIE4x4_DH.sh
```

Generator sinh vào output của **từng run**: default và cả hai localized guesses
cho mọi functional, bao gồm PBE; không giả định PBE chỉ có một basin. Không version
hàng trăm input được sinh lặp từ cùng geometry. `AutoAux` dùng cho fit basis;
kiểm tra độ nhạy DF nếu cần đối chiếu định lượng. Launcher copy orbital đọc vào
đúng tên `merged.gbw` và giữ failed scratch. Collector yêu cầu SCF convergence,
normal termination và **verdict stability cuối** đã nhận diện. Unknown stability
không được chấp nhận tự động; nếu ORCA version có wording khác, cập nhật parser theo
log thực và thêm fixture test. DH chọn reference SCF energy thấp nhất rồi lấy DH energy.
Không có bộ ORCA executable trong environment kiểm chứng này, nên chưa chạy end-to-end ORCA.
Một mình PBE gần Table 1 không chứng nhận mọi functional DH hay lựa chọn basin đều đúng.
