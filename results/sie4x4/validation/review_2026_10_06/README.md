# Kiểm chứng pipeline SIE — 2026-10-06

Đây là kiểm chứng theo trường hợp trên basis nhỏ, **không phải kết quả hoàn chỉnh
SIE4x4 16 điểm ở cùng basis**. Không tính một MUE chung từ các basis khác nhau.
Các phép tính đều dùng charge +1, Nα−Nβ=1, không ép ⟨S²⟩=0.75;
OBDH α=(0.53,0.39), không embedding/CL, second-order BCH mặc định của upstream.

## Kết quả về init

`OBDH_init_comparison.csv` gồm 15 case:

| Phép thử | Basis/grid | Phạm vi | Kết quả |
|---|---|---|---|
| H₂⁺…He | cc-pVDZ/3 | 4 R hữu hạn + DL | HF/UKS-init đều cho OBDH deloc; abs(ΔE) < 1e-9 Eh |
| He₂⁺ | cc-pVDZ/3 | 4 R hữu hạn + DL | 4 R hữu hạn: abs(ΔE) < 1.3e-9 Eh; DL: nhánh HF-init chưa hội tụ OBDH trong 200 vòng, UKS-init có nghiệm deloc |
| (NH₃)₂⁺ | cc-pVDZ/3 | Rₑ | Hai nguồn cho OBDH deloc; abs(ΔE) = 4.63e-9 Eh |
| (NH₃)₂⁺ | STO-3G/2 | Rₑ + DL | Rₑ deloc, DL mixed; hai nguồn chênh < 1.7e-9 Eh |
| (H₂O)₂⁺ | STO-3G/2 | Rₑ + DL | Hai nguồn cho OBDH deloc; abs(ΔE) < 3.7e-10 Eh |

Ở 14 case có kết quả hội tụ từ cả hai nguồn, max abs(ΔE) < 5e-9 Eh.
Một case chỉ có UKS-init hợp lệ được ghi rõ; không dùng energy của HF chưa hội tụ.
Đây là kết quả của các case đã chạy, không khẳng định mọi hệ/basis/parameter sẽ
độc lập với init. Giữ hai nguồn là lựa chọn hợp lý: một nguồn có thể tìm được
nghiệm mà nguồn còn lại chưa hội tụ.

Các bảng này kiểm chứng HF-reference/grid/final-density/strict-convergence fixes;
`run_configs.json` lưu driver hash thực dùng cho từng lượt. Warm-orbital search
được mở rộng sau đó. Vì vậy đây là dữ liệu kiểm chứng các thành phần, không phải
một lượt production duy nhất từ cùng revision driver. `OBDH_candidates.csv` giữ
mọi kết quả OB, kể cả chưa hội tụ, iteration diagnostics, populations và s2;
`OBDH_selected.csv` chỉ chọn các nghiệm hợp lệ. Không tự đổi nhãn mixed thành loc.

## Cùng orbital phải cho cùng OBDH

`same_orbitals_HF_UKS.json`: He₂⁺, cc-pVDZ, grid 2, α=(0.53,0.39), MP2-fit cc-pVDZ-RI.
Truyền cùng orbital UKS vào OBDH theo UKS object hoặc UHF carrier:
energy = −4.988968900349563 Eh cho cả hai, ΔE = 0 và norm Δdensity = 0;
cả hai hội tụ. Unit regression còn kiểm tra không thay đổi orbital của UKS seed.
Đây là kiểm chứng lỗi HF operator đã được sửa, không phải so sánh hai nghiệm
SCF HF/UKS độc lập.

## Density restart và orbital warm start không tương đương

H₂⁺…He, cc-pVDZ/grid 3, DL 10Re và He cách H gần nhất 6 Å:
restart SCF bằng density có thể đi vào nghiệm metastable có hole trên He.
Nghiệm PBE đó có energy khoảng −2.985868421 Eh, charge He ≈ +1 và ⟨S²⟩ ≈ 1.75.
Nó không bị loại chỉ vì spin. Tuy nhiên nó cao hơn nghiệm tìm được qua
orbital warm optimization khoảng **0.491570356 Eh**.

`SCF_orbital_warm.json` ghi nghiệm thấp hơn, với convergence + internal stability:

| Phương pháp | E(DL), Eh | Charge He | Spin populations hai H | ⟨S²⟩ |
|---|---:|---:|---|---:|
| PBE | −3.477438777681402 | ≈ 1.0e-9 | ≈ 0.5000244 / 0.4999756 | ≈ 0.750000001 |
| PBE0 | −3.457396010025311 | ≈ 4.7e-10 | ≈ 0.5000292 / 0.4999708 | ≈ 0.750000000 |

Do đó pipeline mới chuyển cả orbitals và occupations vào Newton cho warm guess,
thay vì chỉ đọc density rồi diagonalize Aufbau. Fragment-average cũng được thử,
nhưng riêng nó không bảo đảm giữ đúng basin trong phép thử này.

Từ E_PBE(Rₑ) ≈ −3.491792913687 Eh, De ≈ 9.01 kcal/mol và signed error ≈ −55.39
kcal/mol, gần xu hướng/số −54.8 của bài ở aug-cc-pVTZ. Đây là một kiểm chứng một
điểm trên basis khác, không chứng nhận tái lập toàn bảng hoặc mọi phương pháp.

## Lượt lặp lại qua driver warm-orbital

`warm_endpoint_*.csv` và `warm_endpoint_config.json` là một lượt riêng của driver
đã mở rộng warm orbitals, H₂⁺…He Rₑ + DL, cc-pVDZ/grid 3, methods PBE/PBE0/OBDH.
Lượt này hoàn tất, không thiếu selected energy, giữ các metastable/failed candidates
trong audit. Các thay đổi guard energy witness/helper-source sau đó được kiểm tra
bằng unit tests; config hash giữ đúng revision thực dùng cho lượt tính.

## Kiểm tra đã chạy và giới hạn

- 10 unit/regression tests đạt: HF carrier không có xc, seed được giữ nguyên,
  full fragment AO block, chia fragment ở mọi hình học, doublet electron counts,
  DL distances/He placement, không dùng S2 gate, không chấp nhận SCF chưa hội tụ
  chỉ vì Hessian ổn định, ORCA final stability/normal termination, statistics subset,
  warm orbitals không bị đổi thành density restart, và không báo cáo một reference cao hơn khi đã có lower-energy determinant witness.
- Kiểm tra SCF thật bằng `tests/check_orbital_warm.py` đã đạt;
  `public_warm_regression.json` giữ energy, gradient, s2 và fragment populations cuối.
- Patch installer: dry-run, áp dụng đúng hash, chạy lại idempotent đã kiểm tra.
- Sinh 700 ORCA inputs (100 case × 7 files) trong output của run; Bash syntax
  được kiểm tra. Chưa chạy ORCA executable nên chưa có end-to-end ORCA validation.
- OB convergence là dE + effective Fia theo solver hiện có; chưa có true OB
  orbital Hessian để chứng nhận variational stability/global minimum.
- Lượt aug-cc-pVDZ chạy các R hữu hạn H₂⁺…He, nhưng bị dừng ở DL vì SCF khó hội tụ;
  không đưa vào MUE hay coi là full run. Lượt cc-pVDZ được dừng sau 11 case hoàn tất
  để chuyển sang kiểm chứng bổ sung warm-orbital và hai hệ lớn trên basis nhỏ hơn.
- Có thể chạy full 20 case/16 De trên HPC bằng driver mới, outdir mới, và tăng
  scf/ob cycles nếu thiếu nghiệm; không nới spin gate hoặc dùng energy chưa hội tụ.

Bài/SI: Bao–Gagliardi–Truhlar, JPCL 2018, DOI 10.1021/acs.jpclett.8b00242,
Table 1, SI Tables S4/S8. Geometry đã đối chiếu; basis nhỏ/grid PySCF và density
fitting là khác biệt chủ động của phép kiểm chứng này.
