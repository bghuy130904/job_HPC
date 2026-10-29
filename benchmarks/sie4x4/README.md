# SIE4x4 v2: phép tính độc lập, kiểm tra neighbor và báo cáo rõ nguồn

## Chạy một lần cho toàn bộ 81 cặp Alpha

Từ root repo, dùng nhánh `workspace/benchmark-layout`:

```bash
bash benchmarks/sie4x4/jobs/obdh/job.sh --submit
```

Không điền từng `ALPHA=(...)` nữa. Slurm array mặc định `0-80%3`: tối đa
ba task chạy đồng thời, một task tính cả bốn hệ và năm hình học tại một Alpha.
Thứ tự: task 0=(0.1,0.1), 8=(0.1,0.9), 9=(0.2,0.1), 80=(0.9,0.9).
Job tổng hợp tự được submit với dependency `afterany`, nên vẫn báo cáo các
điểm thiếu khi một số task thất bại; không đổi thiếu dữ liệu thành năng lượng 0.

```bash
# Chỉ chạy hàng alpha_x=0.1, tối đa hai task đồng thời:
ARRAY_SPEC=0-8%2 bash benchmarks/sie4x4/jobs/obdh/job.sh --submit
# Tạo lại báo cáo tổng hợp mà không chạy phép tính lượng tử:
bash benchmarks/sie4x4/jobs/obdh/job.sh --merge
```

Kết quả mới mặc định ở
`/data/giahuy/Result/OBDH/SIE_DFT/scaling_v2/fixed_pipeline/`.
Từng task dùng `recalc_obdh_(x,c)`; bảng toàn lưới ở `scan_summary/`.
Dữ liệu v2 cũ nằm ngoài thư mục này và không bị sửa, ghi đè hay đổi nhãn.
Có thể đổi `SCAN_ROOT`, `RESULT_ROOT`, `CODE_ROOT`, `INPUT_JSON`, `BASIS` bằng environment.

## Thư mục tạm trên /data, không dùng /scratch của node

Áp dụng cách đặt TMPDIR của `dipole_cccbdb/jobs/ccsd/job.sh`:
`TMP_ROOT=/data/giahuy/pyscf_tmp/sie4x4`. Cả `TMPDIR` (Python/Excel) và
`PYSCF_TMPDIR` được đặt trước khi khởi động Python. Mỗi array/task/attempt có
một thư mục `run.XXXXXXXX` riêng. Log ghi đường dẫn, dung lượng và inode của filesystem.
Trap dọn đúng thư mục tạm vừa tạo cả khi Python trả lỗi. `KEEP_TMP=1` giữ lại
thư mục đó để debug. Không dọn thư mục của task khác hoặc kết quả benchmark.
`TMP_ROOT=/một/ổ/khác` có thể ghi đè vị trí mặc định.

## Ba đường tính có ý nghĩa khác nhau

1. **Kết quả chính từ UHF:** hội tụ SCF UHF tại hình học hiện tại, thử các guess
   hiện tại, loại density trùng, rồi chạy OBDH từ từng nghiệm SCF hợp lệ.
2. **Kết quả chính từ DFT:** hội tụ DFT tại đúng Alpha hiện tại với
   `ax*HF + (1-ax)*B88, (1-ac)*LYP`. Từ cùng một bộ orbital/occupation/orbital
   energies, chạy hai nhánh: OBMP2 một lần và OBDH đến hội tụ.
3. **Kiểm tra neighbor:** lấy orbital OBDH độc lập đã hội tụ tại Alpha lân cận,
   đưa thẳng vào OBDH với Alpha hiện tại. Không hội tụ lại SCF ở đường này.
   `source` chỉ là lịch sử UHF/DFT của orbital; đường tính thực là `neighbor_obdh`.

Neighbor và restart **không được thay thế kết quả chính** trong `selected`,
`errors` hoặc `statistics`. Nếu chỉ neighbor hội tụ, kết quả chính vẫn để trống.
Năng lượng thấp nhất tìm được khi kể cả các phép kiểm tra nằm ở
`diagnostic_best`, với đường khởi tạo được ghi riêng.

Không sửa toán tử, DIIS, điều kiện hội tụ hay source pyCMF. UHF carrier của
orbital DFT chỉ thay cách đánh giá các toán tử HF, **không chạy lại HF SCF**.
Phần DFT được tính tại Alpha mục tiêu trong mọi iteration OBDH.
DFT reference và DFT term bên trong OBDH dùng cùng OB auxiliary basis như
định nghĩa v2 trước đây; không chuyển lại auxiliary basis để ép trùng số v1.

## Định nghĩa một lần và cách ghép cặp

`obdh_1st` là **DFT đã hội tụ + một lần đánh giá correction OBMP2**, không phải
iteration đầu của OBDH bắt đầu từ UHF:

```text
E_first = E_DFT_SCF + alpha_c * E_corr_OBMP2[trạng thái DFT hội tụ]
```

Nếu DFT tìm được hai nghiệm A/B, có các cặp first-A/full-A và first-B/full-B.
`initial_orbital_id` băm cả coefficients, occupations và orbital energies để
kiểm tra đúng cùng trạng thái ban đầu. `paired_valid` đòi hỏi hai lượt hợp lệ
và mã orbital ban đầu giống nhau. `label_changed` báo loc/deloc/mixed thay đổi.
Cùng nhãn **không chứng nhận cùng nhánh**; `same_final_branch_certified=False`.
Không loại cặp đổi nhãn âm thầm: chúng đo tác động tự nhất quán kèm đổi kiểu nghiệm.

`obdh_1st` trong selected dùng nghiệm DFT có năng lượng **DFT** thấp nhất,
không chọn theo năng lượng sau correction. OBDH chính chọn năng lượng thấp
nhất trong các lượt từ SCF độc lập, nên hai dòng selected không nhất thiết
được ghép cùng orbital. Dùng `paired_relaxation`, không trừ tùy ý hai dòng
selected. Bảng tổng hợp `paired_errors` chọn đúng cặp của DFT reference đã
chọn tại từng R/R_DL rồi mới tính năng lượng phân ly và sai số.
`paired_statistics` báo MUE của hai nhánh từ đúng các reference này, chỉ khi
đủ 16 phản ứng. Không thay nó bằng MUE của OBDH được chọn độc lập.

## Neighbor trong array: xác định trước và không truyền lịch sử vô hạn

Mỗi task chọn trước một Alpha kề: giảm alpha_c 0.1 nếu còn trong lưới, nếu
không thì giảm alpha_x, rồi tăng alpha_c/alpha_x. Không chọn nguồn dựa vào
Alpha nào tình cờ chạy xong trước. Nếu nguồn đúng này chưa sẵn sàng, task tự
tính một baseline hỗ trợ trong `neighbor_reference/`, tắt neighbor/restart/probe
để không đệ quy. Nguồn hỗ trợ thất bại không ngăn baseline hiện tại được chạy.

Chỉ export/import **nghiệm OBDH baseline được chọn** của mỗi nguồn, mỗi hình học.
Không nhập toàn bộ orbital neighbor/restart đã tích lũy. Tối đa một OB reference
cho mỗi nguồn từ một thư mục neighbor. Không dùng neighbor làm SCF guess của
baseline; đó là nguyên tắc giữ phép tính chính độc lập với lưới và thứ tự array.
Nghiệm từ neighbor dùng cùng hình học, basis, charge/spin; cấu hình toán tử,
ngưỡng và phiên bản driver cũng được đối chiếu trước khi tự tái sử dụng.

## Đọc log và Excel

Log tổng mỗi task: `sie4x4_ax_x_ac_c.txt`; log đầy đủ từng hệ/R:
`SYSTEM__POINT.log`. Mỗi block có CURRENT TARGET, CALCULATION ROLE,
INITIALIZATION PATH, mã orbital ban đầu, định nghĩa năng lượng, điều kiện
chạy và REPORTED RESULT. Raw energy của pure OBMP2 bên trong one-shot không
phải năng lượng one-shot được báo cáo; log in phép ghép DFT + scaled correction.
Probe năng lượng giữ orbital energies cố định được ghi riêng, không trộn
vào các iteration OBDH hội tụ. `dRMS` là DIIS residual, không phải density RMS.

| Sheet | Đọc để biết gì? |
|---|---|
| selected | Năng lượng chính từ SCF hiện tại. Thiếu thì để trống, không cứu bằng neighbor. |
| diagnostic_best | Năng lượng thấp nhất tìm được khi kể cả neighbor/restart, không dùng cho MUE chính. |
| statistics / errors | MUE đủ 16 điểm / sai số có dấu từng hệ và R. |
| branch_map | Nhãn, điện tích/spin từng fragment, determinant S2 và đường khởi tạo của kết quả chính. |
| candidates | Toàn bộ lượt SCF/OB, trạng thái và lỗi; không chỉ các nghiệm được chọn. |
| paired_relaxation | So first/full từ cùng orbital DFT; ghi đổi loại nghiệm và giới hạn diễn giải. |
| energy_components / frozen_derivatives | Các thành phần năng lượng và Dx/Dc tại trạng thái được xét. |
| neighbor_comparison / ob_restarts | Đối chiếu neighbor với baseline / nhiễu baseline rồi hội tụ lại. |
| C1_C4 | Độ đầy đủ của phép tính và phần yêu cầu khoa học còn thiếu. |
| readme / config | Định nghĩa cột, đơn vị, các điều kiện và hash dùng để kiểm chứng. |

## C1-C4: không đồng nhất chạy xong với chứng nhận khoa học

- C1: populations/type được ghi; restart thử nghiệm baseline; **chưa có full OB
  stability analysis**. Probe không thay thế Hessian của năng lượng.
- C2: localized SCF guesses thuộc baseline; direct neighbor là phép kiểm tra
  riêng. Giữ cả hai kết quả. Không diễn giải nhãn nguồn neighbor là fresh SCF.
- C3: `scan_summary/signed_errors_ax_0.7.png` và `...0.9.png` vẽ sai số có dấu
  cho bốn hệ tại bốn R. Đường nối ngắt khi thiếu dữ liệu hoặc đổi nhãn tại R/R_DL.
  Dự đoán dấu của PDF là giả thuyết để kiểm tra, không phải tiêu chí loại số liệu.
- C4: `slope_comparison` đối chiếu Dx/Dc tại (0.5,0.4) với central finite
  differences từ Alpha kề; đánh dấu stencil đổi nhãn hoặc thiếu dữ liệu.
  Đây không phải đồng nhất thức đạo hàm tự hợp; cùng nhãn không chứng nhận cùng nhánh.

`execution_complete` chỉ nói phép kiểm tra khả dụng đã chạy được.
`pdf_complete=False` không bị đổi thành True chỉ vì có số: stability OB và
phân tích khoa học về nhánh còn thiếu. `--confirm-c1-c4` yêu cầu dữ liệu cho
các phép kiểm tra khả dụng; exit nonzero khi baseline/diagnostic execution thiếu,
không khẳng định hoàn thành toàn bộ PDF. Job tổng hợp vẫn chạy sau các task lỗi.

## Resume và kiểm chứng

Resume chỉ nhận đúng config/input/code/neighbor snapshot. Đổi code/cấu hình
phải chọn SCAN_ROOT mới. Với cùng snapshot, `--retry-incomplete` chạy lại các
case thiếu kết quả chính; case chính đã hoàn tất được tái sử dụng. JSON giữ
lỗi tài nguyên riêng; thiếu dữ liệu không tự động được gọi là không hội tụ số học.

```bash
python -m unittest discover -s benchmarks/sie4x4/tests -p test_scaling_v2.py -v
python -m unittest discover -s benchmarks/sie4x4/tests -p test_scaling_v2_summary.py -v
python -m unittest discover -s benchmarks/sie4x4/tests -p test_scaling_jobs.py -v
```

Các phép thử nhỏ là kiểm chứng implementation, không phải kết quả scaling HPC
và không được đưa vào Result như dữ liệu benchmark.

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



