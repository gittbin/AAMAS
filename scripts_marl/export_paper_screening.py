"""Export audited development screening without representing it as final evidence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts_marl.audit_paper_messages import audit_screening
from experiments_paper_completion.io import write_json


def main():
    out=ROOT / "outputs_marl/paper_v4/screening"
    audit=audit_screening(out)
    d=json.loads((out / "report/results.json").read_text(encoding="utf-8"))
    lines=["% Audited development screen; never pooled with final confirmation.","\\begin{resulttable}\\centering\\small", "\\caption{Development screen: three fresh matched seeds, 150,528 transitions per run, 30 assessment scenarios. Checkpoints locked before assessment.}","\\begin{tabular}{lrrr}\\toprule Variant &Return &On--off &Normal--zero\\\\\\midrule"]
    labels={"no_comm":"No communication","comm_reference":"Communication reference","comm_payload":"Enriched local payload","comm_rotating_router":"Rotating router ties"}
    for condition,label in labels.items():
        mean=d["estimates"][condition+"/selected"]["net_return"]["mean"]
        controls=d["communication_controls"].get(condition)
        effects=[f"{controls[name]['mean']:+.3f}" for name in ("comm_off","content_zero")] if controls else ["--","--"]
        lines.append(f"{label} &{mean:.3f} &{effects[0]} &{effects[1]}\\\\")
    lines += ["\\bottomrule\\end{tabular}\\label{tab:v4screening}\\end{resulttable}"]
    for condition,label in (("comm_payload","enriched payload"),("comm_rotating_router","rotating tie router")):
        e=d["comparisons_vs_comm_reference"][condition]
        lines.append(f"The {label} changed return relative to the reference by {e['mean']:+.3f}, CI $[{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]$, positive in {e['positive_training_pairs']}/3 matched seeds.")
    lines.append("Neither branch met all predeclared advancement criteria: higher selected return than the reference in at least two of three pairs, with positive mean on--off and normal--zero effects. No new payload/router variant was advanced. These development comparisons are exploratory and do not establish final superiority, equivalence, or a unique mechanism for failure.")
    dest=ROOT / "AAMAS_rewrite_complete (1)/generated/screening_v4.tex"
    dest.write_text("\n\n".join(lines)+"\n",encoding="utf-8")
    write_json(out / "report/manuscript_export.json",{"audit_passed":audit["audit_passed"],"scope":"development only","artifact":dest.relative_to(ROOT).as_posix()})


if __name__=="__main__": main()
