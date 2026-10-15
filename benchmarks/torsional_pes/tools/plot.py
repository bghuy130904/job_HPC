import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from matplotlib.offsetbox import AnchoredText

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Ưu tiên dùng Times New Roman
    "mathtext.fontset": "stix",     # Giúp các ký hiệu toán học (H_2, He_2^+) đồng bộ với Times New Roman
    "font.size": 14,                # Kích thước font cơ bản
    "axes.labelsize": 16,           # Kích thước nhãn của các trục (X, Y)
    "axes.titlesize": 18,           # Kích thước tiêu đề của từng đồ thị con
    "legend.fontsize": 10,          # Kích thước chữ trong hộp chú thích (legend)
    "xtick.labelsize": 22,          # Kích thước các con số trên trục X
    "ytick.labelsize": 22,          # Kích thước các con số trên trục Y
    "axes.labelweight": "bold",     # Tự động in đậm toàn bộ nhãn trục X và Y
    "axes.titleweight": "bold"      # Tự động in đậm toàn bộ tiêu đề
})


# ── 1. Tăng tiếp độ dày nét và kích thước marker để nhìn siêu rõ trên A4 ──
SHELL_STYLES = {
    1: dict(color="#1f77b4", linestyle="-",  marker="o", markersize=8, linewidth=3.0, label="0th shell"),
    2: dict(color="#ff7f0e", linestyle="--", marker="s", markersize=8, linewidth=3.0, label="1st shell"),
    3: dict(color="#2ca02c", linestyle=":",  marker="^", markersize=8, linewidth=3.0, label="2nd shell"),
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_xy(data):
    phi = [pt["phi"] for pt in data["scan"]]
    # Giữ nguyên E - E_ref đã lưu; không đổi về E(180), không thay điểm lỗi bằng 0.
    e = [pt["e_rel_kcal"] if pt.get("status") in (None, "converged", "completed")
         and pt.get("e_rel_kcal") is not None else np.nan for pt in data["scan"]]
    return np.array(phi), np.array(e)

# ── Đường dẫn đến các file JSON ─────────────────────────────────────────────
# Đường dẫn INPUT lấy trực tiếp từ OUTPUT_DIR của jobs/{obdh,mp2}/input.sh.
OBDH_DIR = Path("/data/giahuy/Result/OBDH_in_DFT/bench_torsional_pes")
MP2_DIR = Path("/data/giahuy/Result/MP2_in_DFT/bench_torsional_pes")
# Trên HPC: /home/giahuy/Code/job/results/torsional_pes.
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "results" / "torsional_pes"
VALIDATION_DIR = OUTPUT_DIR / "validation"

parser = argparse.ArgumentParser(description="Plot torsional PES using the original figure layout.")
parser.add_argument("--validation", action="store_true",
                    help="Chỉ vẽ 12 JSON cũ trong results/torsional_pes/validation.")
args = parser.parse_args()
if args.validation:
    OBDH_DIR = MP2_DIR = VALIDATION_DIR

FILES1 = {ns: OBDH_DIR / f"ethanol_pes_ns{ns}.json" for ns in (1, 2, 3)}
FILES2 = {ns: OBDH_DIR / f"1-butanol_pes_ns{ns}.json" for ns in (1, 2, 3)}
FILES3 = {ns: MP2_DIR / f"ethanol_mp2_pes_ns{ns}.json" for ns in (1, 2, 3)}
FILES4 = {ns: MP2_DIR / f"1-butanol_mp2_pes_ns{ns}.json" for ns in (1, 2, 3)}

missing = [path for files in (FILES1, FILES2, FILES3, FILES4)
           for path in files.values() if not path.is_file()]
if missing:
    raise SystemExit("Thiếu file dữ liệu; không tự thay bằng validation:\n" +
                     "\n".join(str(path) for path in missing))

# ── Đọc dữ liệu ─────────────────────────────────────────────────────────────
datasets1 = {ns: extract_xy(load_json(fp)) for ns, fp in FILES1.items()}
datasets2 = {ns: extract_xy(load_json(fp)) for ns, fp in FILES2.items()}
datasets3 = {ns: extract_xy(load_json(fp)) for ns, fp in FILES3.items()}
datasets4 = {ns: extract_xy(load_json(fp)) for ns, fp in FILES4.items()}

# ── 2. Hàm vẽ Panel với Fontsize và khoảng đệm (pad) siêu to ────────────────
def plot_panel(ax, datasets, title=None, ylabel=None, show_legend=False, method_label=None):
    for ns, (phi, e) in datasets.items():
        style = SHELL_STYLES[ns]
        ax.plot(phi, e,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=style["markersize"],
                linewidth=style["linewidth"],
                label=style["label"],
                markerfacecolor=style["color"],
                markeredgewidth=0.8,
                markeredgecolor="white")

    # Tăng fontsize và đẩy khoảng cách chữ ra xa để chống dính
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold", labelpad=15)
    if title is not None:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    ax.set_xlim(-5, 185)
    ax.set_xticks(range(0, 181, 40))

    # ── TĂNG CỠ SỐ TRÊN 2 TRỤC X, Y ──
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # ── THÊM Ô GHI CHÚ PHƯƠNG PHÁP (GIỐNG LEGEND) ──
    if method_label:
        at = AnchoredText(method_label, loc="upper right",
                          prop=dict(fontsize=14, fontweight="bold"),
                          frameon=True)
        at.patch.set_edgecolor("gray")
        at.patch.set_linewidth(1.2)
        at.patch.set_alpha(0.9)
        at.patch.set_boxstyle("square,pad=0.4")  # Tạo khung vuông vắn đồng bộ với legend
        ax.add_artist(at)

    # Đưa legend xuống vị trí giữa bên phải (center right) để tránh đè nét vẽ phía trên
    if show_legend:
        ax.legend(fontsize=14, frameon=True, framealpha=0.9,
                  edgecolor="gray", fancybox=False,
                  loc="center right")

# ── 3. Giảm tiếp figsize để tăng tỷ lệ hiển thị của chữ và số ──────────────
# Đổi từ (14, 10.5) xuống (12, 9.5). Tránh dùng constrained_layout gây lỗi đè supxlabel.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

datasets_mol1 = datasets1
datasets_mol2 = datasets2
datasets_mol3 = datasets3
datasets_mol4 = datasets4

# Vẽ hàng 1 (OBDH)
# Vẽ hàng 1 (OBDH)
plot_panel(axes[0, 0], datasets_mol1,
           title="Ethanol",
           ylabel="Relative Energy\n(kcal/mol)",  # Đổi thành \n để xuống dòng gọn gàng
           show_legend=False,
           method_label="sub-OBDH")  # Thêm nhãn phương pháp cho hàng 1

plot_panel(axes[0, 1], datasets_mol2,
           title="1-Butanol",
           show_legend=True,  # Ô này chứa legend ở vị trí center right
           #method_label="OBDH"
           )

# Vẽ hàng 2 (MP2)
plot_panel(axes[1, 0], datasets_mol3,
           ylabel="Relative Energy\n(kcal/mol)",  # Đổi thành \n để xuống dòng gọn gàng
           show_legend=False,
           method_label="MP2-in-DFT")   # Thêm nhãn phương pháp cho hàng 2

plot_panel(axes[1, 1], datasets_mol4,
           show_legend=False,
           #method_label="MP2"
           )

# ── 4. Xử lý vị trí supxlabel an toàn tuyệt đối bằng tight_layout ──────────
# rect=[0, 0.08, 1, 1] giữ lại 8% không gian trống hoàn toàn ở đáy hình cho supxlabel
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Đặt dòng chữ dùng chung tại tọa độ y=0.03 nằm gọn trong khoảng trống vừa tạo
fig.supxlabel("C2-C1-OH dihedral (°)", fontsize=18, fontweight="bold", y=0.03)

# Xuất file
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
stem = "torsional_pes_validation" if args.validation else "torsional_pes"
output_pdf = OUTPUT_DIR / f"{stem}.pdf"
plt.savefig(output_pdf, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Đã lưu hình: {output_pdf}")
