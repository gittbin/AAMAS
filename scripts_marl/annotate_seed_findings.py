"""Attach seed audit warnings while preserving original numerical evidence."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.paper_seed_audit import audit_seed_split, SEED_WARNING


def main():
    campaign = ROOT / "outputs_marl/paper_v2/campaign"
    report = campaign / "report"
    snapshot = report / "log_analysis/original_reports_before_seed_check"
    snapshot.mkdir(parents=True, exist_ok=True)
    names = ["audit.json", "results_summary.json", "results.md", "tables_full.tex",
             "KET_QUA_KIEM_TRA.md", "KET_QUA_TUNG_SEED.md", "heldout_comparison.pdf", "heldout_comparison.png"]
    for name in names:
        if (report / name).exists() and not (snapshot / name).exists():
            shutil.copy2(report / name, snapshot / name)
    split = audit_seed_split(campaign, range(40000, 40050))
    assert not split["evaluation_disjoint_from_training"]
    for name in ["results_summary.json", "audit.json"]:
        path = report / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["seed_split_audit"] = split
        data["strict_final_test_valid"] = False
        data["interpretation_warning"] = SEED_WARNING
        if name == "audit.json":
            data["artifact_integrity_passed"] = True
            data["audit_passed"] = False
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    warning = (
        "> **Cập nhật kiểm tra seed:** scenario đánh giá `40000` đã xuất hiện trong huấn luyện seed 0 và 4; "
        "scenario development `10000` đã xuất hiện trong huấn luyện seed 2. Các số liệu 50 scenario bên dưới "
        "vẫn là dữ liệu chạy thật, nhưng bộ này chưa hoàn toàn held-out. Kiểm tra trước đã bỏ sót tính rời nhau "
        "của seed. Bỏ `40000` chỉ là kiểm tra độ nhạy; phải dành bộ final mới sau khi chọn cải tiến. "
        "Xem [phân tích log và hướng cải thiện](log_analysis/PHAN_TICH_LOG_VA_CAI_THIEN.md)."
    )
    for name in ["KET_QUA_KIEM_TRA.md", "KET_QUA_TUNG_SEED.md", "results.md"]:
        path = report / name
        content = path.read_text(encoding="utf-8")
        if "**Cập nhật kiểm tra seed:**" not in content:
            first, rest = content.split("\n", 1)
            path.write_text(first + "\n\n" + warning + "\n" + rest, encoding="utf-8")
    path = report / "tables_full.tex"
    content = path.read_text(encoding="utf-8")
    content = content.replace(
        "Executed held-out results: mean [95\\% nested bootstrap interval]. Five independent trained policies per condition and 50 common held-out scenarios in the final study.",
        "Original evaluation results: mean [95\\% nested bootstrap interval]. Five trained policies and 50 common scenarios per primary condition. Scenario 40000 overlaps training replicates 0 and 4; this is not a fully unseen final test.")
    path.write_text(content, encoding="utf-8")
    print("Seed warnings attached; original reports preserved in", snapshot)


if __name__ == "__main__":
    main()
