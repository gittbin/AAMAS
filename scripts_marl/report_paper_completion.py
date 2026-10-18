"""Audit final raw evidence, report all outcomes, and export only verified findings."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts_marl.audit_paper_messages import rows, trace, accounting_matches, require, close
from scripts_marl.run_paper_confirmation import sha
from experiments_paper_completion.io import write_json
from scripts_marl.report_paper_v2 import nested_interval, exact_test, holm

METRICS = ("net_return", "min_user_rate", "sum_rate", "illumination_db", "outage_fraction", "reconfiguration", "requested", "delivered", "return_before_message_cost", "return_at_default_price")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(out, lock):
    from scripts_marl.run_paper_completion import ensure_gate, select_graphs
    manifest = ensure_gate(out, lock)
    p = lock["protocol"]
    ids = list(range(p["reserved_final_seed_start"], p["reserved_final_seed_start"] + p["reserved_final_scenarios"]))
    protected = set(ids + p["development_seeds"] + p["assessment_seeds"])
    all_resets, datasets = {}, {}
    transitions = cells = episodes = traces = 0
    trace_metrics = ("min_user_rate", "sum_rate", "illumination_db", "outage_fraction", "requested", "eligible", "delivered", "budget_dropped", "graph_rejected", "reconfiguration", "violation_before", "violation_after")
    def validate_trace(path, eps, expected_ids):
        records = trace(path, eps, expected_ids)
        for i, ep in enumerate(eps):
            block = records[i*100:(i+1)*100]
            for metric in trace_metrics:
                require(close(ep[metric], math.fsum(float(r[metric]) for r in block)/100), "Trace/episode " + metric)
                require(close(ep["post25_"+metric], math.fsum(float(r[metric]) for r in block[25:])/75), "Post-freeze metric " + metric)
            require(close(ep["post25_return"], math.fsum(float(r["net_reward"]) for r in block[25:])), "Post-freeze return")
            for key, column in (("return_before_message_cost","reward_before_message_cost"),("return_at_default_price","reward_at_default_price")):
                require(close(ep[key], math.fsum(float(r[column]) for r in block)), "Reward accounting")
        return records
    for condition in p["conditions"]:
        for seed in p["training_seeds"]:
            train = out / "train" / condition / f"seed_{seed}"
            status = read(train / "status.json")
            require(status["complete"] and status["training_complete"] and status["actual_transitions"] == p["steps_per_run"], "Training coverage")
            require(status["source_sha256"] == lock["source_sha256"], "Training source")
            entry = manifest["selected"][f"{condition}/seed_{seed}"]
            for kind, key in (("best","selected_checkpoint_sha256"),("final","final_checkpoint_sha256")):
                require(sha(train / f"checkpoint_{kind}.pt") == entry[key] == status[f"checkpoint_{kind}_sha256"], "Checkpoint hash")
            require(sha(train / "selection.csv") == entry["selection_log_sha256"], "Selection log changed")
            selected = max(rows(train / "selection.csv"), key=lambda r: (float(r["eval_return_mean"]), -int(r["global_step"])))
            require(int(selected["global_step"]) == entry["selected_step"] == status["best_step"] and close(selected["eval_return_mean"], status["best_development_return"]), "Development selection rule")
            actual = [int(r["scenario_seed"]) for r in rows(train / "episode_seeds.csv")]
            require(len(actual) == len(set(actual)) and not set(actual)&protected, "Reset uniqueness/disjointness")
            require(seed not in all_resets or set(actual) == all_resets[seed], "Matched resets")
            all_resets[seed] = set(actual)
            for row in rows(train / "train.csv"):
                require(all(math.isfinite(float(v)) for v in row.values()), "Nonfinite training")
                cells += len(row)
            transitions += status["actual_transitions"]
            folder = out / "evaluation" / condition / f"seed_{seed}"
            data = read(folder / "results.json")
            require(data["complete"] and data["checkpoint_lock"] == entry and data["source_sha256"] == lock["source_sha256"] and data["scenario_ids"] == ids, "Evaluation provenance")
            expected = {"normal"}
            if condition == "full_nocomm": expected |= {"frozen25","frozen10","static_learned"} | {"csi_"+v for v in p["csi_feature_interventions"]}
            if condition == "full_comm" or condition.startswith("candidate_"): expected |= {"comm_off","content_zero"}
            if condition == "full_comm":
                expected |= {"comm_random","comm_forced","content_random","content_permute"}
                expected |= {"graph_"+x for x in ("empty","full","ring","bs_star","selected","harmful")}
                expected |= {"graph_harmful_content_"+x for x in ("zero","random","permute")}
            if condition in ("full_nocomm","feedforward","masked_csi"):
                expected |= {f"tau_{v}" for v in p["tau_grid"]} | {f"delay_{v}" for v in p["delay_grid"]} | {"dynamics_"+v for v in ("static","slow","fast")}
                if condition == "full_nocomm": expected |= {"dynamics_"+v+"_frozen25" for v in ("static","slow","fast")}
            require(set(data["treatments"]) == expected, "Treatment coverage")
            normal = harmful = None
            for treatment, item in data["treatments"].items():
                eps = item["episodes"]
                selected_ids = ids[:p["exploratory_final_scenarios"]] if treatment.startswith(("tau_","delay_","dynamics_")) else ids
                records = validate_trace(folder / f"{treatment}.csv", eps, selected_ids)
                if treatment == "normal": normal = records
                if treatment == "graph_harmful": harmful = records
                if treatment.startswith("content_"): accounting_matches(normal, records)
                if treatment.startswith("graph_harmful_content_"): accounting_matches(harmful, records)
                datasets.setdefault(f"{condition}/{treatment}", []).append((seed, eps))
                episodes += len(eps); traces += len(records)
            if condition == "full_nocomm":
                for prefix in ("", "dynamics_static_", "dynamics_slow_", "dynamics_fast_"):
                    a = "normal" if not prefix else prefix[:-1]
                    b = prefix+"frozen25"
                    require(all(x["prefix25_sha256"] == y["prefix25_sha256"] for x,y in zip(data["treatments"][a]["episodes"],data["treatments"][b]["episodes"])), "Adaptive/frozen prefix")
    require(transitions == p["total_training_transitions"], "Transition total")
    sets = list(all_resets.values())
    require(all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i)), "Independent training blocks")
    graph = read(out / "graph_selection.json")
    # Replay the deterministic selector in an isolated report directory, preserving
    # the immutable original artifact even if the verification fails.
    graph_check = out / "report/graph_selection_verification"
    graph_check.mkdir(parents=True, exist_ok=True)
    (graph_check / "graph_development").mkdir(exist_ok=True)
    for seed in p["training_seeds"]:
        origin = out / "graph_development" / f"seed_{seed}/results.json"
        d = read(origin)
        require(d["complete"] and set(d["graphs"]) == {str(g) for g in range(64)}, "Graph enumeration")
        require(d["checkpoint_sha256"] == manifest["selected"][f"full_comm/seed_{seed}"]["selected_checkpoint_sha256"], "Topology provenance")
        for g, eps in d["episodes_by_graph"].items():
            require([e["scenario_seed"] for e in eps] == p["development_seeds"][:p["topology"]["development_scenarios"]], "Topology development IDs")
            require(all(math.isfinite(v) for e in eps for v in e.values() if isinstance(v,(float,int))), "Topology nonfinite")
            for metric, mean in d["graphs"][g].items(): require(close(mean,np.mean([e[metric] for e in eps])), "Topology aggregation")
        target = graph_check / "graph_development" / f"seed_{seed}/results.json"
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(origin,target)
    select_graphs(graph_check, p)
    require(read(graph_check / "graph_selection.json") == graph, "Development graph selector")
    pairs = ("power_ris","power_ma","ris_ma")
    ranking = {c:float(np.mean([read(out / "train" / c / f"seed_{s}/status.json")["best_development_return"] for s in p["training_seeds"]])) for c in pairs}
    require(read(out / "best_pair_selection.json")["selected_pair"] == max(pairs,key=ranking.get), "Best pair selection")
    baseline = read(out / "baselines/results.json")
    require(baseline["complete"] and set(baseline["references"]) == {"noop","random","heuristic","online_optimizer","ramp_knee","ramp_comm_extreme","ramp_sensing_extreme"}, "Baseline coverage")
    for name, item in baseline["references"].items():
        records = validate_trace(out / "baselines" / f"{name}.csv", item["episodes"], ids)
        datasets["baseline/"+name] = [(None,item["episodes"])]
        episodes += len(item["episodes"]); traces += len(records)
    timing = read(out / "baselines/latency.json")
    require(set(timing) == {"full_nocomm","full_comm","centralized","online_optimizer"}, "Latency coverage")
    for d in timing.values():
        require(d["n"] == p["latency"]["measurements"], "Timing count")
        for name, values in d["samples"].items():
            require(len(values) == d["n"] and all(math.isfinite(v) and v>=0 for v in values), "Timing samples")
            require(close(d["summary"][name]["median"],np.median(values)) and close(d["summary"][name]["p95"],np.percentile(values,95)), "Timing summaries")
    return datasets, {"audit_passed":True, "source_sha256":lock["source_sha256"], "training_runs":len(p["conditions"])*len(p["training_seeds"]), "training_transitions":transitions, "training_numeric_cells":cells, "episodes":episodes, "trace_rows":traces, "nonfinite_cells":0, "checkpoint_hashes_valid":True, "actual_train_dev_final_disjoint":True, "matched_blocks_valid":True, "fixed_send_corruption_valid":True, "freeze_prefix_valid":True, "development_graph_selection_reproduced":True, "locked_before_final":True, "scope":p["stage"], "time_unix":time.time()}


def matrix(datasets, key, metric):
    data = sorted(datasets[key],key=lambda x: -1 if x[0] is None else x[0])
    require(all([e["scenario_seed"] for e in eps] == [e["scenario_seed"] for e in data[0][1]] for _,eps in data), "Scenario matrix alignment")
    return np.asarray([[e[metric] for e in eps] for _,eps in data])


def analyse(datasets, p):
    summaries, per_seed = {}, []
    for key, data in sorted(datasets.items()):
        means = {m:{"mean":float(matrix(datasets,key,m).mean()),"training_sd":float(matrix(datasets,key,m).mean(axis=1).std(ddof=1)) if len(data)>1 else None} for m in METRICS}
        if key.endswith("/normal") or key.startswith("baseline/") or key == "full_nocomm/static_learned":
            for m in ("net_return","min_user_rate","illumination_db"):
                means[m].update(nested_interval(matrix(datasets,key,m),repetitions=p["bootstrap_repetitions"],seed=p["statistics_seed"]))
        summaries[key] = {"training_policies":0 if key.startswith("baseline/") else len(data), "scenarios_per_policy":len(data[0][1]), "metrics":means}
        for seed, eps in data:
            per_seed.append({"condition":key.split("/")[0],"treatment":key.split("/")[1],"seed":seed,"scenarios":len(eps),**{m:float(np.mean([e[m] for e in eps])) for m in METRICS}})
    primary = []
    for h in p["primary_hypotheses"]:
        left,right = matrix(datasets,h["left"],h["metric"]),matrix(datasets,h["right"],h["metric"])
        effect = nested_interval(left,right,paired=True,repetitions=p["bootstrap_repetitions"],seed=p["statistics_seed"])
        differences = left.mean(axis=1)-right.mean(axis=1)
        primary.append(dict(h,**effect,p_exact=exact_test(left,right,paired=True),wins=int((differences>0).sum()),paired_seed_effects=differences.tolist()))
    for item, adjusted in zip(primary,holm([h["p_exact"] for h in primary])): item["p_holm"] = adjusted
    exploratory = []
    specs = [("full_comm/normal","full_comm/content_zero","net_return"),("full_comm/normal","full_comm/content_random","net_return"),("full_comm/normal","full_comm/content_permute","net_return"),("full_nocomm/normal","feedforward/normal","net_return"),("full_nocomm/normal","masked_csi/normal","net_return"),("full_comm/graph_harmful","full_comm/graph_harmful_content_zero","net_return")]
    specs += [("full_nocomm/dynamics_"+v,"full_nocomm/dynamics_"+v+"_frozen25","post25_min_user_rate") for v in ("static","slow","fast")]
    for left,right,metric in specs:
        effect=nested_interval(matrix(datasets,left,metric),matrix(datasets,right,metric),paired=True,repetitions=p["bootstrap_repetitions"],seed=p["statistics_seed"])
        exploratory.append(dict(left=left,right=right,metric=metric,**effect,scope="exploratory; no primary adjusted test"))
    j = lambda key: summaries[key]["metrics"]["net_return"]["mean"]
    online,static = j("baseline/online_optimizer"),j("full_nocomm/static_learned")
    gap = online-static
    performance = {c:{"absolute_gap_to_online":j(c+"/normal")-online,"raw_return_ratio_to_online":j(c+"/normal")/online if abs(online)>1e-8 else None,"gain_recovery":(j(c+"/normal")-static)/gap if gap>1e-8 else None} for c in ("full_nocomm","full_comm","centralized")}
    return {"summaries":summaries,"per_seed":per_seed,"primary":primary,"exploratory":exploratory,"performance_relative_to_online":performance,"gain_recovery_denominator":gap}


def plot(report, data, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(11,8))
    keys = ["baseline/noop","baseline/ramp_knee","baseline/heuristic","full_nocomm/normal","full_comm/normal","centralized/normal","baseline/online_optimizer"]
    for ax,metric in zip(axes[0],("net_return","min_user_rate")):
        values = [data["summaries"][k]["metrics"][metric] for k in keys]
        means=np.array([v["mean"] for v in values]); ci=np.array([v["ci95"] for v in values])
        ax.errorbar(range(len(keys)),means,yerr=np.maximum(0,np.stack([means-ci[:,0],ci[:,1]-means])),fmt="o",capsize=3)
        ax.set_xticks(range(len(keys)),[k.replace("baseline/","").replace("/normal","") for k in keys],rotation=35,ha="right")
        ax.set_ylabel(metric); ax.grid(alpha=.25)
    for c in ("full_nocomm","feedforward","masked_csi"):
        for ax,grid,kind in ((axes[1,0],p["tau_grid"],"tau"),(axes[1,1],p["delay_grid"],"delay")):
            ax.plot(grid,[data["summaries"][f"{c}/{kind}_{v}"]["metrics"]["min_user_rate"]["mean"] for v in grid],"o-",label=c)
            ax.set_xlabel(kind); ax.set_ylabel("Min-user rate (exploratory)"); ax.grid(alpha=.25)
    axes[1,1].legend(fontsize=8); fig.tight_layout()
    fig.savefig(report / "complete_study.pdf"); fig.savefig(report / "complete_study.png",dpi=180); plt.close(fig)


def markdown(report,data,p,audit_result,timing):
    text = ["# Kết quả đợt xác nhận đầy đủ", "",f"Scope: {p['stage']}. {len(p['training_seeds'])} seed/nhánh, {p['steps_per_run']:,} transitions/run; final {p['reserved_final_scenarios']} scenario/policy. Audit đạt, không NaN/Inf. Đơn vị kiểm định là cặp training seed; scenario không được coi là seed độc lập.","", "| Điều kiện | Return | Min-rate | Illumination dB | Outage | Delivered |","|---|---:|---:|---:|---:|---:|"]
    for key,d in data["summaries"].items():
        if key.endswith("/normal") or key.startswith("baseline/") or key=="full_nocomm/static_learned":
            text.append("| "+key+" | "+" | ".join(f"{d['metrics'][m]['mean']:.4f}" for m in ("net_return","min_user_rate","illumination_db","outage_fraction","delivered"))+" |")
    text += ["","## Bảy giả thuyết chính đã khóa","","| Giả thuyết | Hiệu ứng | CI95 | p exact | p Holm | Số cặp dương |","|---|---:|---|---:|---:|---:|"]
    for h in data["primary"]:
        text.append(f"| {h['name']} ({h['metric']}) | {h['mean']:+.4f} | [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}] | {h['p_exact']:.5f} | {h['p_holm']:.5f} | {h['wins']}/{len(p['training_seeds'])} |")
    text += ["","Không có ý nghĩa thống kê không chứng minh tương đương. Giá, budget, nội dung tin, độ trễ, dynamics và topology là phân tích thăm dò. Hãy đọc trade-off sensing/outage và kết quả từng seed trước khi diễn giải.","",f"Audit: {audit_result['training_runs']} train runs, {audit_result['training_transitions']:,} transitions; {audit_result['episodes']:,} episode, {audit_result['trace_rows']:,} dòng trace. Chọn checkpoint/static/topology hoàn tất trước khi mở final; không chọn lại bằng kết quả final.","","Latency đo decision kernels trên CPU một thread, gồm BD/power riêng; không đo end-to-end triển khai hoặc đường truyền mạng. Số đo, mẫu gốc và scope nằm trong baselines/latency.json.","","Kết quả đầy đủ: results.json; từng seed: per_seed.csv; biểu đồ: complete_study.pdf; audit: audit_final.json. Submission readiness chỉ được đánh giá sau build PDF và kiểm tra metadata."]
    (report / "KET_QUA_HOAN_THIEN.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def export_paper(out, data, p, audit_result, timing):
    paper = ROOT / "AAMAS_rewrite_complete (1)"
    dest = paper / "generated"; dest.mkdir(exist_ok=True)
    def val(key,m): return data["summaries"][key]["metrics"][m]["mean"]
    def table(caption,label,entries,metrics,head,wide=False):
        env="resultwide" if wide else "resulttable"
        lines=[f"\\begin{{{env}}}","\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",f"\\caption{{{caption}}}","\\begin{tabular}{l"+"r"*len(metrics)+"}\\toprule",head+"\\\\\\midrule"]
        for name,key in entries: lines.append(name+" &"+" &".join(f"{val(key,m):.3f}" for m in metrics)+"\\\\")
        return "\n".join(lines+["\\bottomrule\\end{tabular}\\label{"+label+"}",f"\\end{{{env}}}"])
    entries=[("Initial layout","baseline/noop"),("Archive knee","baseline/ramp_knee"),("Archive rate extreme","baseline/ramp_comm_extreme"),("Archive sensing extreme","baseline/ramp_sensing_extreme"),("Estimated-CSI heuristic","baseline/heuristic"),("Random","baseline/random"),("Policy-selected static","full_nocomm/static_learned"),("Decentralized, no messages","full_nocomm/normal"),("Decentralized, messages","full_comm/normal"),("Centralized recurrent PPO","centralized/normal"),("Online optimizer","baseline/online_optimizer")]
    sections=[f"% Audited completion source {audit_result['source_sha256']}","\\subsection{Controllers and adaptation}",table(f"Controller comparison: {len(p['training_seeds'])} independent matched training blocks, {p['steps_per_run']:,} transitions per run, and {p['reserved_final_scenarios']} common untouched final scenarios. Means over policies and scenarios; nonlearning references have no training replicates.","tab:main",entries,("net_return","min_user_rate","illumination_db","outage_fraction","delivered"),"Controller &Return &Min-rate &Illumination &Outage &Delivered",True)]
    sections += ["Uncertainty uses a paired nested bootstrap over training blocks and shared scenario IDs (10,000 draws). The seven prespecified primary tests use two-sided exact sign flips over block means with Holm correction. Table~\\ref{tab:primary} includes every primary result, irrespective of direction.","\\begin{resultwide}\\centering\\scriptsize", "\\caption{Primary differences: left minus right. Intervals are pointwise 95\\% nested-bootstrap intervals; $p_H$ adjusts all seven primary tests.}\\begin{tabular}{lrrrr}\\toprule Comparison &Difference &CI lower &CI upper &$p_H$\\\\\\midrule"]
    labels={"adaptation":"Adaptive--frozen25 (post-freeze rate)","full_vs_ma":"Full--MA (rate)","full_vs_power_ris":"Full--power+RIS (rate)","full_vs_power_ma":"Full--power+MA (rate)","full_vs_ris_ma":"Full--RIS+MA (rate)","comm_vs_nocomm":"Comm-trained--no-comm (return)","communication_on_vs_off":"Messages on--off (return)"}
    for h in data["primary"]: sections.append(f"{labels[h['name']]} &{h['mean']:+.3f} &{h['ci95'][0]:+.3f} &{h['ci95'][1]:+.3f} &{h['p_holm']:.4f}\\\\")
    sections += ["\\bottomrule\\end{tabular}\\label{tab:primary}\\end{resultwide}"]
    for h in data["primary"]:
        wording="statistically supported" if h["p_holm"]<.05 else "not statistically confirmed"
        sections.append(f"{labels[h['name']]} was {wording}: $\\Delta={h['mean']:+.3f}$, CI $[{h['ci95'][0]:+.3f},{h['ci95'][1]:+.3f}]$, $p_H={h['p_holm']:.4f}$ ({h['wins']}/{len(p['training_seeds'])} positive blocks).")
    sections += ["\\subsection{Actuator contribution}",table("All actuator subsets retrained without messages under the same budget.","tab:actuator",[(name,c+"/normal") for name,c in (("Power","power"),("RIS","ris"),("MA","ma"),("Power+RIS","power_ris"),("Power+MA","power_ma"),("RIS+MA","ris_ma"),("Full","full_nocomm"))],("net_return","min_user_rate","illumination_db"),"Subset &Return &Min-rate &Illumination")]
    best=read(out / "best_pair_selection.json")["selected_pair"].replace("_","+")
    sections.append(f"The highest-return pair on development was {best}; it was locked before final. All three pair comparisons remain in the primary test family. A larger team alone does not establish coordination.")
    sections += ["\\subsection{Information, memory, and dynamics}","\\begin{figure*}[t]\\centering\\includegraphics[width=.92\\textwidth]{figures/complete_study.pdf}\\caption{Controller uncertainty (top) and exploratory CSI/delay sensitivity (bottom). Curves use the first 30 final scenarios per policy; their shared scenarios are not independent training replicates.}\\label{fig:complete}\\end{figure*}"]
    for h in data["exploratory"]:
        if "feedforward" in h["right"] or "masked_csi" in h["right"] or "dynamics" in h["left"]:
            label=h["left"].replace("_","\\_")+" minus "+h["right"].replace("_","\\_")
            sections.append(f"Exploratory {label}: $\\Delta={h['mean']:+.3f}$, CI $[{h['ci95'][0]:+.3f},{h['ci95'][1]:+.3f}]$ for {h['metric'].replace('_',' ')}.")
    sections += ["Flat delay curves do not establish memory compensation. All robustness interventions freeze checkpoint selection; shifted scenarios are reported separately from the primary nominal comparisons.","\\subsection{Communication and prices}",table("Retrained delivery-price/budget controls. Requests and deliveries are per control interval; returns retain each condition's stated price.","tab:prices",[(n,c+"/normal") for n,c in (("Budget1/free","price_free"),("Budget1/default","full_comm"),("Budget1/tenfold","price_tenfold"),("Budget3/free","budget3_free"),("Budget3/default","budget3_default"),("Budget3/tenfold","budget3_tenfold"),("Request price/default","request_price"))],("net_return","min_user_rate","requested","delivered"),"Price/budget &Return &Min-rate &Requests &Delivered")]
    for h in data["exploratory"]:
        if h["left"] == "full_comm/normal":
            mode=h["right"].split("/")[1].replace("_"," ")
            sections.append(f"Normal minus {mode}, with send actions, timing, identities and delivery accounting fixed: return $\\Delta={h['mean']:+.3f}$, CI $[{h['ci95'][0]:+.3f},{h['ci95'][1]:+.3f}]$ (exploratory).")
    sections += ["The budget-three model also has a larger inbox input. Budget changes therefore jointly change delivery capacity and actor input size; they are not a pure bandwidth intervention. Price comparisons within each budget share architecture and exact initialization.","\\input{generated/diagnostics.tex}","The earlier eight-pair confirmation is a separate, already observed study (comm-trained minus no-comm return $+1.765$, CI $[-1.839,6.172]$; on minus off $-0.570$, CI $[-1.074,0.516]$). The subsequent three-seed development screen gave returns 49.752 for the communication reference, 47.797 for enriched payloads and 47.192 for a rotating tie router; neither candidate passed the locked advancement criteria. These results motivated the fresh study and are not pooled with its final data.","\\subsection{Topology and retraining}",table("Graphs selected on development before final. These are interventions on fixed full-graph policies; retraining is separate.","tab:graphs",[(n,"full_comm/graph_"+g) for n,g in (("Empty","empty"),("Full","full"),("Ring","ring"),("BS star","bs_star"),("Selected front","selected"),("Harmful","harmful"))],("min_user_rate","illumination_db","delivered"),"Graph &Min-rate &Illumination &Delivered")]
    graph=read(out / "graph_selection.json")
    sections.append(f"The development front contained {len(graph['pareto_front'])} of 64 directed graphs; selected graph ID {graph['selected_graph']} was retrained with the full matched budget. Its final return was {val('selected_graph/normal','net_return'):.3f} and min-rate {val('selected_graph/normal','min_user_rate'):.3f}. The graph front is conditional on the frozen policies and is not a global control optimum.")
    sections += ["\\subsection{Computation and reproducibility}","\\begin{resulttable}\\centering\\scriptsize\\caption{Uncontended CPU decision kernels, batch 1, one Torch thread. Median/P95 milliseconds; acquisition, channel construction, feature construction and transport excluded.}\\begin{tabular}{lrr}\\toprule Controller &Median &P95\\\\\\midrule"]
    for name in ("full_nocomm","full_comm","centralized","online_optimizer"):
        d=timing[name]["summary"]["decision_kernels_ms"]
        sections.append(f"{name.replace('_',' ')} &{d['median']:.3f} &{d['p95']:.3f}\\\\")
    sections += ["\\bottomrule\\end{tabular}\\end{resulttable}",f"Raw evidence passed checks on {audit_result['training_runs']} training runs, {audit_result['training_transitions']:,} transitions and {audit_result['trace_rows']:,} evaluation trace rows, with no nonfinite metrics. Actual training, development and final scenario IDs were disjoint; all paired blocks shared training resets. Per-seed outcomes, all exploratory treatments, topology artifacts and component timing samples are included in the reproducibility report. Timing measures simulated decision kernels rather than deployed end-to-end latency."]
    (dest / "completion.tex").write_text("\n\n".join(sections)+"\n",encoding="utf-8")
    shutil.copy2(out / "report/complete_study.pdf",paper / "figures/complete_study.pdf")
    source=paper / "main.tex"; original=source.read_text(encoding="utf-8")
    backup=out / "report/manuscript_before_completion.tex"
    if not backup.exists(): backup.write_text(original,encoding="utf-8")
    start=original.index("\\section{Results}"); end=original.index("\\section{Discussion and Limitations}",start)
    new=original[:start]+"\\section{Results}\n\\label{sec:results}\n\\input{generated/completion.tex}\n\n"+original[end:]
    new=re.sub(r"\\fillresult\{Add supported empirical interpretations.*?\}","",new)
    empirical=" ".join(f"{labels[h['name']]} changed by {h['mean']:+.3f} (Holm $p={h['p_holm']:.4f}$)." for h in data["primary"] if h["name"] in ("adaptation","comm_vs_nocomm","communication_on_vs_off"))
    new=new.replace("\\textbf{[Empirical summary intentionally blank: insert verified adaptation, coordination, communication, robustness, and latency findings after final evaluation.]}",f"A fresh {len(p['training_seeds'])}-block study tests seven prespecified contrasts on {p['reserved_final_scenarios']} final scenarios. "+empirical)
    new=new.replace("\\textbf{[Final empirical conclusions intentionally blank: insert only verified findings and scope.]}",empirical+" These findings are conditional on the codebooks, information boundaries and tested training budget; measured latency covers decision kernels only.")
    oldstatus="\\noindent\\textbf{Draft status.}"
    if oldstatus in new:
        left=new.index(oldstatus); right=new.index("\n\\else",left)
        new=new[:left]+"\\noindent\\textbf{Draft status.} All complete-study results have passed artifact checks. Submission metadata and page limits are assessed separately; this is a reading preview."+new[right:]
    new=new.replace("The empirical contribution will be stated from the completed results.","Primary results are reported with multiplicity correction; exploratory findings retain their stated scope.")
    source.write_text(new,encoding="utf-8")
    return build_paper(paper,out / "report")


def build_paper(paper,report):
    tool=ROOT / ".tools-paper/tectonic/tectonic.exe"
    import os
    temp=ROOT / ".tmp-paper"; temp.mkdir(exist_ok=True)
    cache=ROOT / ".tools-paper/tectonic-cache"; cache.mkdir(parents=True,exist_ok=True)
    build_env=dict(os.environ,TEMP=str(temp),TMP=str(temp),TECTONIC_CACHE_DIR=str(cache))
    builds={}
    for name in ("main","preview"):
        result=subprocess.run([str(tool),"--keep-logs","--keep-intermediates",name+".tex"],cwd=paper,env=build_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
        (report / f"latex_{name}.log").write_text(result.stdout,encoding="utf-8")
        builds[name]={"success":result.returncode==0,"returncode":result.returncode}
    if builds["preview"]["success"]: shutil.copy2(paper / "preview.pdf",paper / "paper_progress.pdf")
    metadata=bool(re.search(r"\\acmSubmissionID\{(?!XXXX\})[^}]+\}",(paper / "main.tex").read_text(encoding="utf-8")))
    pages=None; reference_page=None; content_page=None
    if builds["main"]["success"]:
        from pypdf import PdfReader
        pdf=PdfReader(paper / "main.pdf"); pages=len(pdf.pages)
        reference_page=next((i+1 for i,page in enumerate(pdf.pages) if re.search(r"(?mi)^\s*REFERENCES\s*$",page.extract_text())),None)
        if (paper / "main.aux").exists():
            marker=re.search(r"\\newlabel\{lastcontentpage\}\{\{[^}]*\}\{(\d+)\}",(paper / "main.aux").read_text(encoding="utf-8"))
            if marker: content_page=int(marker.group(1))
        shutil.copy2(paper / "main.pdf",report / "paper_official_format.pdf")
    page_check=content_page is not None and content_page<=8
    result={"builds":builds,"submission_id_present":metadata,"pdf_pages":pages,"references_start_page":reference_page,"last_content_page":content_page,"content_within_eight_pages":page_check,"automated_submission_checks_passed":all(d["success"] for d in builds.values()) and metadata and page_check,"submission_ready":False,
            "remaining_manual_checks":["Confirm submission metadata and registered abstract constraints","Review figure/table placement and anonymity","Review official track requirements before submission"]}
    write_json(report / "MANUSCRIPT_STATUS.json",result)
    return result


def finish(out,lock,write_paper=False):
    report=out / "report"; report.mkdir(exist_ok=True)
    datasets,evidence = audit(out,lock)
    p=lock["protocol"]
    result=analyse(datasets,p)
    result.update(protocol=p,source_sha256=lock["source_sha256"],scope=p["stage"],audit_passed=True,final_scenarios_opened=True)
    write_json(report / "audit_final.json",evidence)
    write_json(report / "results.json",result)
    with (report / "per_seed.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result["per_seed"][0])); writer.writeheader(); writer.writerows(result["per_seed"])
    timing=read(out / "baselines/latency.json")
    markdown(report,result,p,evidence,timing); plot(report,result,p)
    if write_paper:
        require(p["stage"]!="pipeline_smoke", "Smoke evidence must not overwrite manuscript")
        status=export_paper(out,result,p,evidence,timing)
        write_json(report / "MANUSCRIPT_STATUS.json",status)
        package_supplement(out)
    return result


def package_supplement(out):
    """Small anonymous code/summary package; bulky traces/checkpoints stay local."""
    import zipfile
    report=out / "report"
    files=list((out / "source_snapshot").rglob("*.py"))+list((out / "source_snapshot").rglob("*.json"))+list((out / "source_snapshot").rglob("*.yaml"))
    files += [out / "requirements_resolved.txt",out / "protocol_locked.json",out / "execution_protocol.json",out / "selected_checkpoints_locked.json",out / "graph_selection.json",out / "best_pair_selection.json",out / "baselines/latency.json"]
    files += [report / name for name in ("results.json","per_seed.csv","audit_final.json","complete_study.pdf","KET_QUA_HOAN_THIEN.md")]
    files += list((out / "graph_development").glob("*/results.json"))
    def anonymize(value):
        if isinstance(value,dict): return {k:anonymize(v) for k,v in value.items() if k not in ("pid","log","machine","hostname")}
        if isinstance(value,list): return [anonymize(v) for v in value]
        if isinstance(value,str): return value.replace(str(ROOT),"<PROJECT>").replace(str(ROOT).replace('\\','/'),"<PROJECT>")
        return value
    path=report / "supplement_anonymous.zip"
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for f in files:
            name=f.relative_to(out).as_posix()
            if f.suffix==".json": archive.writestr(name,json.dumps(anonymize(read(f)),indent=2,allow_nan=False))
            else: archive.write(f,name)
        archive.write(ROOT / "docs/AI_ASSISTANCE_DISCLOSURE.md","AI_ASSISTANCE_DISCLOSURE.md")
        archive.writestr("README.md","Anonymous reproducibility material. Source/configuration snapshots, independent seed protocol, locked selections, statistical summaries and per-seed results are included. Raw step traces and full policy checkpoints are retained locally and omitted for size. Reproduction uses scripts_marl/run_paper_completion.py with a fresh output directory; the primary campaign does not retune on final outcomes. The manuscript and authors' disclosure record require final author review.\n")
    require(path.stat().st_size<=25*1024*1024,"Supplement exceeds 25 MB")
    write_json(report / "supplement_status.json",{"path":path.name,"bytes":path.stat().st_size,"within_25mb":True,"sha256":sha(path),"raw_traces_and_checkpoints_included":False,"anonymity_manual_review_remaining":True})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--write-paper",action="store_true")
    args=parser.parse_args(); finish(args.out.resolve(),read(args.out / "execution_protocol.json"),args.write_paper)
