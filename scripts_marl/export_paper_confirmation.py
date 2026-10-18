"""Insert audited confirmation evidence into the manuscript without filling other studies."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.run_paper_confirmation import verify_sources


def main():
    campaign = ROOT / "outputs_marl/paper_v3/confirmation_v2"
    verify_sources(json.loads((campaign / "protocol_locked.json").read_text()))
    audit = json.loads((campaign / "report/audit_final.json").read_text())
    if not audit["audit_passed"]:
        raise ValueError("Only audited confirmation may be exported")
    results = json.loads((campaign / "report/results.json").read_text())
    estimates = results["estimates"]
    lines = [r"% Generated from audited paper-v3/confirmation_v2, not development screening.",
             r"\begin{resultwide}", r"\centering\small",
             r"\caption{Locked communication confirmation: eight fresh matched training pairs, 400,384 transitions per run, and 50 common final scenarios per policy. Checkpoints were selected on 30 development scenarios before final evaluation. Means are over policies and scenarios; paired uncertainty is reported in the text.}",
             r"\begin{tabular}{lrrrr}\toprule",
             r"Training / evaluation &Return &Min-rate &Requests &Delivered\\\midrule"]
    choices = [("no_comm", "selected", "No communication"), ("comm_reference", "selected", "Communication, normal"),
               ("comm_reference", "comm_off", "Communication, messages off"),
               ("comm_reference", "content_zero", "Communication, zero payload / fixed sends")]
    for condition, treatment, label in choices:
        values = estimates[f"{condition}/{treatment}"]
        requests = []
        for seed in range(30, 38):
            d = json.loads((campaign / "evaluation" / condition / f"seed_{seed}/results.json").read_text())
            requests += [e["requested"] for e in d["treatments"][treatment]["episodes"]]
        requested = sum(requests) / len(requests)
        lines.append(f"{label} &{values['net_return']['mean']:.3f} &{values['min_user_rate']['mean']:.3f} &{requested:.3f} &{values['delivered']['mean']:.3f}" + r"\\")
    lines += [r"\bottomrule\end{tabular}\label{tab:comm}", r"\end{resultwide}"]
    a, b = results["primary_comparisons"]
    zero = results["exploratory_selected_minus_content_zero"]["net_return"]
    def interval(item):
        return f"[{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]"
    lines += ["", f"Communication-trained policies exceeded no-communication policies by {a['mean']:+.3f} in mean net return "
              f"(95\\% paired nested-bootstrap interval {interval(a)}; two-sided exact sign-flip $p={a['exact_p']:.5f}$; Holm-adjusted $p={a['holm_p']:.5f}$). "
              "The sign was positive in four of eight matched training pairs; the measured advantage was not statistically confirmed.", "",
              f"Switching messages on rather than off within the same trained communication policy changed net return by {b['mean']:+.3f} "
              f"(95\\% interval {interval(b)}; exact $p={b['exact_p']:.5f}$; Holm-adjusted $p={b['holm_p']:.5f}$). "
              "Disabling messages increased return for seven of eight policies. The two prespecified tests formed one Holm family. "
              "These results neither establish equivalence nor show communication is universally harmful.", "",
              f"With every send action replayed and sender identities, availability, timing, delivery counts and cost preserved, "
              f"normal minus zero-payload return was {zero['mean']:+.3f} (95\\% interval {interval(zero)}). "
              "This exploratory content intervention did not establish a consistent content benefit. "
              "The final scenarios were disjoint from actual training and development resets. No checkpoint was selected again using final outcomes. "
              "These communication results are a separate protocol from the earlier diagnostic baseline study and do not complete the actuator, memory or topology comparisons."]
    path = ROOT / "AAMAS_rewrite_complete (1)/generated/confirmation.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
