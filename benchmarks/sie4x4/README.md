# SIE4x4 v1 và v2

Hai driver được đặt tên rõ theo phiên bản: `tools/run_sie_v1.py` và
`tools/run_sie_v2.py`. Mỗi lần gọi vẫn nhận một cặp `--alpha`.
`tools/run_sie.py` chỉ là file chuyển tiếp cho import/lệnh cũ; không chứa
bản sao pipeline. Các file ORCA được giữ nguyên.

Driver mới: `methods/obdh/calc_OBDH_scaling.py` -> `tools/run_sie_v2.py`.
`obdh_1st` được định nghĩa là **hội tụ DFT base trước, rồi đánh giá một lần
OBMP2 correlation trên orbital/occupation/orbital energy DFT đó**:

```text
E_obdh_1st = E_DFT_SCF + alpha_c * E_corr_OBMP2[state_DFT]
```

Nó khác OBDH từ UHF chạy `--ob-cycles 1`. One-shot có
`status=evaluated_one_shot`, `evaluation_complete=True`, `ob_converged=False`;
không dùng `--accept-unconverged`. DFT và OB dùng cùng OB auxiliary basis để
DFT term được hội tụ đúng với operator được đánh giá sau đó. UHF init giữ
`--scf-aux` cũ. Trong v2, source `uhf` và `dft` được báo cáo riêng.

Sửa `ALPHA=(0.9 0.6)` trong job rồi submit:

```bash
bash benchmarks/sie4x4/jobs/obdh/job.sh --submit
```

Giữ partition/CPU/RAM, module, venv và đường dẫn HPC cũ. Job tự cập nhật
Alpha trong input CLI, job name, tên log và thư mục
`/data/giahuy/Result/OBDH/SIE_DFT/scaling_v2/recalc_obdh_(0.9,0.6)`.
Không cần tạo input geometry riêng cho từng Alpha. Có thể đặt `ALPHA_X`,
`ALPHA_C`, `OUTPUT_DIR`, `CODE_ROOT`, `INPUT_JSON`, `BASIS` qua environment.

```bash
# Thử nhanh định nghĩa mới (standard chỉ dành cho phép thử này):
python benchmarks/sie4x4/methods/obdh/calc_OBDH_scaling.py \
  --outdir /tmp/sie_first_check --alpha 0.9 0.6 --basis 6-31g --grid 1 \
  --systems He2_plus --points R_1.0 dissociation_limit \
  --methods obdh obdh_1st --scf-policy standard --threads 1

# C2: thử orbital của ô Alpha đã chạy xong, gồm direct OB continuation:
ALPHA_X=0.9 ALPHA_C=0.7 bash benchmarks/sie4x4/jobs/obdh/job.sh --submit \
  --neighbor-dir '/data/giahuy/Result/OBDH/SIE_DFT/scaling_v2/recalc_obdh_(0.9,0.6)'
```

`config.json`, log/JSON từng điểm và `orbitals/*.npz` lưu nguồn dữ liệu cho
resume/continuation. Không trộn kết quả khác config/input/code/neighbor vào
cùng outdir. SIE dùng supermolecule tại R_DL, charge=1, spin=1 như v1.

Workbook tự ghi các sheet `statistics`, `selected`, `convergence`, `errors`,
`paired_relaxation`, `energy_components`, `frozen_derivatives`, `candidates`,
`iterations`, `orbital_probes`, `ob_restarts`, `neighbor_comparison`, `C1_C4`, `readme`, `config`. Năng lượng thiếu để trống;
MUE_16 chỉ có khi đủ 16 điểm. `paired_relaxation` ghép đúng cùng DFT seed.
`energy_components` chứa riêng HF exchange, B88 exchange, LYP correlation,
DFT và OBMP2 correction. Iterations ghi populations, S2, gap, amplitude và
mẫu số ở amplitude khác không. `diis_residual_rms` không phải density RMS.

Mặc định có hai energy-direction probes/OB candidate hội tụ, giữ orbital
energies/occupations và tính lại amplitudes/density energy. Đây **không phải
full OB Hessian hoặc chứng nhận stability**; SCF stability và OB convergence
được ghi tách biệt. Dùng `--ob-probe-directions 0` để bỏ các probe bổ sung.
C5/C6 không được thêm thành bộ benchmark mới trong thay đổi này.

Chi tiết so sánh phiên bản nằm trong repo Result:
`OBDH/SIE_DFT/SCALING_V1_V2.md`. Các kết quả nhỏ trong kiểm chứng không phải
MUE benchmark và không được đưa vào repo Result như kết quả scaling_v2.

## Job mặc định kiểm tra C1–C4

Job OBDH hiện bật rõ `--confirm-c1-c4`, không chỉ chạy lại năng lượng:

| Mục | Thực hiện trong job |
|---|---|
| C1 | Hai nhiễu occupied/virtual ưu tiên chuyển charge/spin giữa fragment; chạy lại OBDH đến hội tụ tại cùng Alpha; so năng lượng và mật độ. Giữ thêm các energy-direction probes riêng. |
| C2 | Tự tìm ô lân cận đã hoàn tất (bước Alpha mặc định 0.1). Nếu chưa có, tự tính một ô phụ trong `neighbor_reference/` rồi dùng orbital đó. Không cần tự truyền neighbor-dir. |
| C3 | Tính sai số có dấu ở từng R, ghi rõ nghiệm/mốc R_DL; không dùng 0 thay kết quả thiếu. |
| C4 | DFT hội tụ -> OBMP2 một lần; phân rã năng lượng và so với OBDH hội tụ từ cùng seed DFT. |

Chỉ sửa `ALPHA=(...)` rồi `bash .../job.sh --submit`. `NEIGHBOR_STEP`
trong job mặc định 0.1. Ô phụ có `run_role=neighbor_reference`, tắt C1/C2 để
tránh đệ quy, và không được coi là một kết quả grid đã xác nhận C1–C4.
Nếu thư mục Alpha đã có run v2 cũ không bật đủ C1–C4, job tự dùng hậu tố
`_C1_C4` để không trộn dữ liệu.
Các orbital nguồn được cố định khi bắt đầu run; resume không tự đổi sang ô
nguồn vừa xuất hiện sau đó. Cấu hình/code/neighbor khác cần outdir mới.

Sheet `C1_C4` ghi từng mục đã được thực hiện hay bị chặn; `tested` chỉ nghĩa
là đã thực hiện phép kiểm tra. Đọc `ob_restarts` để biết có nghiệm thấp hơn
hoặc mật độ khác xuất hiện không, và `neighbor_comparison` để so baseline
với neighbor. Nếu thiếu năng lượng hoặc không làm đủ các mục, job kết thúc
nonzero và ghi rõ mục chưa xác nhận, không báo đã kiểm chứng đầy đủ.

C1 ở đây là perturb-and-reconverge của phương trình OB; nó không phải full
Hessian hay chứng nhận cực tiểu năng lượng. Nếu chỉ gọi Python cho một phép
thử riêng, có thể tắt phần tự chuẩn bị neighbor hoặc restarts bằng CLI;
launcher kiểm chứng C1–C4 không tắt các mục này mặc định.

## Kiểm chứng implementation

- Bốn test trong `tests/test_scaling_v2.py`: định nghĩa DFT + một OBMP2
  correction, giữ orbital/epsilon (cho phép đổi thứ tự cột), phân biệt với
  UHF-init một vòng, reference thấp nhất thất bại không bị thay bằng reference
  cao hơn, và kiểm tra geometry của neighbor.
- Đã chạy He2+ tại R0/R_DL với 6-31G/grid 1, gồm multiguess, full OBDH hai
  source, one-shot, phân rã năng lượng và directional probes.
- Đã chạy one-shot cho (H2O)2+ tại R0/R_DL với 6-31G/grid 1; direct neighbor
  continuation cho He2+ ở Alpha lân cận; và He2+ tại R0 với aug-cc-pVDZ/grid 4.
- Workbook, trạng thái thiếu dữ liệu, partial MUE, resume và tên job tự sinh
  đã được kiểm tra. Kiểm tra submit dùng sbatch giả lập; không submit job HPC.

Đây là kiểm chứng luồng tính/ghi dữ liệu, không phải kết quả SIE4x4 đầy đủ
hoặc bằng chứng tái hiện số liệu của paper.

## Quy trình scaling_v1 (giữ để đối chiếu)

# SIE4x4: guesses, nghiệm và benchmark

`tools/run_sie_v1.py` là protocol chung. `methods/dft/calc_DFT.py` và
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
4. Helper UHF/matched UKS (và PBE0 khi chạy PBE) vẫn được tạo khi chỉ yêu cầu DFT để thăm dò basin; chỉ các method yêu cầu được báo cáo. Warm guesses chạy Newton trực tiếp từ orbitals/occupations; việc chỉ đọc lại density rồi diagonalize Aufbau có thể làm mất basin và đưa lỗ điện tích lên He. Theo hướng instability cũng dùng Newton với orbital rotation trả về. Không dừng sớm vì hai energy gần nhau. Chỉ deduplicate density đã hội tụ/ổn định
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
nghiệm OB chưa hội tụ. Nếu một warm determinant hợp lệ có energy SCF thấp hơn mọi
nghiệm hội tụ của source >1e-6 Eh, reference search được đánh dấu chưa hoàn tất:
không dùng reference cao hơn để báo cáo DFT/UMP2/DH reaction energy. Các stable
seeds vẫn được giữ để OB thăm dò. Đây là energy witness, không phải spin/charge gate. Nhãn cuối tính từ orbital density `ob.gamma`, không từ seed.

Spin populations và charges là Mulliken **tổng theo fragment**. Nhãn mô tả dùng
f = |sA−sB|/(|sA|+|sB|): f≥0.8 → loc; f≤0.2 → deloc; còn lại mixed.
Đây là quy ước chẩn đoán phụ thuộc basis/population, không là tiêu chí loại nghiệm.
⟨S²⟩ được tính cho determinant từ orbitals cuối, không là correlated spin observable.
Luôn đọc các population liên tục cùng energy thay vì chỉ đọc nhãn loc/deloc.

Mặc định α=(0.5,0.4) được **giữ nguyên** từ benchmark cũ. `dh_matched` dùng
E_KS + 0.4 E_MP2 trên KS orbitals, làm đối chứng cùng hệ số. Không gọi bộ này
B2PLYP chuẩn. B2PLYP có (0.53,0.27): có thể chạy một outdir riêng với
`--alpha 0.53 0.27`. Đây là biến thể nghiên cứu bổ sung, không tự ý đổi định nghĩa OBDH.

## Cài và chạy

Pipeline dùng API hiện có của pyCMF; không yêu cầu sửa source thư viện.
`hf_orbital_carrier()` trong driver sao chép orbitals/occupations/energies sang UHF
mà không chạy lại SCF. Grid của UKS được tạo bên trong solver được cấu hình tạm
qua `lib.temporary_env(Grids, level=...)` cho từng lần gọi đồng bộ rồi khôi phục.
Diagnostics vòng lặp được đọc từ log của solver. Không thay đổi phương trình,
convergence hoặc các benchmark khác dùng pyCMF.
Đã kiểm tra với upstream `af3614dcd89b78f68b2d5912ddd33a2d277f631a` nguyên bản.

Từ root `job_HPC`, với environment có PySCF, NumPy, SciPy, pandas, openpyxl, pyCMF:

```bash
python -m unittest discover -s benchmarks/sie4x4/tests -v
python benchmarks/sie4x4/tools/run_sie_v1.py \
  --outdir results/sie4x4/validation/avdz-grid4 \
  --basis aug-cc-pVDZ --grid 4 --threads 4
```

Thử nhanh trước khi full benchmark:

```bash
python benchmarks/sie4x4/tools/run_sie_v1.py \
  --outdir results/sie4x4/validation/quick \
  --basis aug-cc-pVDZ --grid 3 --threads 2 \
  --systems He2_plus --points R_1.0 dissociation_limit
```

SLURM: giữ cấu hình HPC gốc: module Python/ORCA, venv, partition, CPU/RAM,
đường dẫn output/error dưới `/data/giahuy/Result`, scratch `/scratch/$SLURM_JOB_ID`
và format script. Chỉ đường dẫn driver/input trong repo được đổi theo layout mới;
lệnh Python thêm `--outdir`, `--threads` để gọi pipeline SIE. Các script ngoài SIE
giữ nguyên nội dung như trước khi tổ chức thư mục; chưa migrate absolute paths của chúng.

```bash
sbatch benchmarks/sie4x4/jobs/dft/job.sh
bash benchmarks/sie4x4/jobs/obdh/job.sh --submit
```

Truyền lựa chọn qua CLI, ví dụ `bash benchmarks/sie4x4/jobs/obdh/job.sh --submit
--basis cc-pVDZ --grid 3 --points R_1.0 dissociation_limit`.
`--methods`, `--systems`, `--points`, `--alpha` và các tolerance được lưu trong config.
Thread count giữ theo cấu hình 10 threads trong launcher gốc. Bản DF mặc định dùng
`def2-universal-jkfit` cho SCF và auxiliary MP2-fit tự chọn riêng cho OB/UMP2.

Từng case có `.log` và `.json` checkpoint, rồi tổng hợp CSV + `SIE4x4.xlsx`.
Resume chỉ khi config, input, driver và pyCMF hash khớp. Đổi basis/alpha/grid/code
phải dùng outdir mới. Đường dẫn point giữ nguyên `R_1.0`, `R_1.25`, ... không bị
mất phần thập phân. Job exit nonzero khi có selected energy thiếu.
`MUE_16` chỉ hiện nếu đủ cả 16 De hợp lệ; `MUE_available` kèm n_valid chỉ mô tả
subset; `common_statistics` dùng giao điểm hợp lệ giữa các method được yêu cầu.

## ORCA là đối chứng tùy chọn

```bash
sbatch benchmarks/sie4x4/jobs/orca/job_SIE4x4_DH.sh
```

Generator sinh vào output của **từng run**: default và cả hai localized guesses
cho mọi functional, bao gồm PBE; không giả định PBE chỉ có một basin. Không version
hàng trăm input được sinh lặp từ cùng geometry. `AutoAux` dùng cho fit basis;
kiểm tra độ nhạy DF nếu cần đối chiếu định lượng. Launcher copy orbital đọc vào
đúng tên `merged.gbw`; script gốc sao lưu scratch của run lỗi sang `_debug` trước khi dọn. Collector yêu cầu SCF convergence,
normal termination và **verdict stability cuối** đã nhận diện. Unknown stability
không được chấp nhận tự động; nếu ORCA version có wording khác, cập nhật parser theo
log thực và thêm fixture test. DH chọn reference SCF energy thấp nhất rồi lấy DH energy.
Không có bộ ORCA executable trong environment kiểm chứng này, nên chưa chạy end-to-end ORCA.
Một mình PBE gần Table 1 không chứng nhận mọi functional DH hay lựa chọn basin đều đúng.

## Kết quả kiểm chứng đi kèm

Xem `results/sie4x4/validation/review_2026_10_06/README.md`, các CSV init/candidates,
config hashes và hai JSON kiểm chứng operator/warm orbitals. Đây là subset nhiều
basis, không phải full SIE4x4 MUE. Có thể chạy kiểm tra vật lý warm guess riêng:

```bash
python benchmarks/sie4x4/tests/check_orbital_warm.py
```



