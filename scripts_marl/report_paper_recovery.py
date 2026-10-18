"""Full frozen-protocol audit plus explicit provenance of the perfect-CSI erratum."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts_marl import report_paper_completion as original
from experiments_paper_completion.io import write_json
from scripts_marl.run_paper_completion import sha

DISCLOSURE = ("After final opening, the original evaluator failed in the exploratory perfect-CSI grid "
    "(feed-forward block 305, scenario 12000003, step 0): float32 inverse-squared estimated gains "
    "overflowed during power normalization. An isolated numerical erratum uses float64 normalization "
    "only when the original decoder is nonfinite, retaining the same gain floor, codebook, power floor "
    "and physical model. All 30 perfect-CSI policy cells were reevaluated uniformly. Finite original "
    "traces were reproduced exactly apart from wall-clock timings. Primary and nominal results retain "
    "their original evaluator and raw traces. Checkpoints, seeds and development selections were not changed.")


def finish_recovery(out, origin, lock):
    ledger=original.read(out/'numerical_erratum_locked.json')
    for name,digest in ledger['correction_source_files'].items():
        original.require(sha(ROOT/name)==digest,'Correction source changed after lock')
    reused=[]
    for source in (origin/'evaluation').rglob('*.csv'):
        if source.name=='tau_0.0.csv': continue
        target=out/source.relative_to(origin)
        original.require(sha(source)==sha(target),'Original final trace was altered')
        reused.append({'path':source.relative_to(origin).as_posix(),'sha256':sha(source)})
    for source in (origin/'train').rglob('*'):
        if source.is_file() and source.name!='status.json':
            original.require(sha(source)==sha(out/source.relative_to(origin)),'Training artifact altered')
    for condition in ('full_nocomm','feedforward','masked_csi'):
        for seed in lock['protocol']['training_seeds']:
            d=original.read(out/f'erratum_evidence/{condition}/seed_{seed}.json')
            original.require(d['complete'] and d['finite_original_trace_comparison_passed'], 'Erratum trace comparison failed')
    result=original.finish(out,lock,write_paper=False)
    audit=original.read(out/'report/audit_final.json')
    provenance={'disclosure':DISCLOSURE,'correction_lock_sha256':sha(out/'numerical_erratum_locked.json'),
                'primary_evaluator_unchanged':True,'original_failure_preserved':True,
                'scope':'exploratory tau=0.0 grid only; 30 policy cells',
                'reused_original_trace_files':len(reused),'original_trace_identity_passed':True,
                'correction_events':sum(original.read(f)['correction_events_count'] for f in (out/'erratum_evidence').glob('*/seed_*.json'))}
    result['numerical_erratum']=provenance; audit['numerical_erratum']=provenance
    write_json(out/'report/original_trace_identity.json',{'passed':True,'files':reused})
    write_json(out/'report/results.json',result); write_json(out/'report/audit_final.json',audit)
    md=out/'report/KET_QUA_HOAN_THIEN.md'; text=md.read_text(encoding='utf-8')
    md.write_text(text.replace('\n\n','\n\n**Numerical erratum / công khai sửa lỗi sau khi mở final:** '+DISCLOSURE+'\n\n',1),encoding='utf-8')
    # Defer compilation until the manuscript explicitly includes the correction.
    export=types.FunctionType(original.export_paper.__code__,dict(original.export_paper.__globals__,
        build_paper=lambda paper,report: None),argdefs=original.export_paper.__defaults__)
    export(out,result,lock['protocol'],audit,original.read(out/'baselines/latency.json'))
    paper=ROOT/'AAMAS_rewrite_complete (1)'; tex=paper/'generated/completion.tex'
    content=tex.read_text(encoding='utf-8')
    content=content.replace('Curves use the first 30 final scenarios per policy;',
        'The perfect-CSI point uses the disclosed overflow correction. Curves use the first 30 final scenarios per policy;')
    # "Untouched" is no longer accurate for the corrected exploratory cells.
    content=content.replace('common untouched final scenarios','common final scenarios, selected independently of training')
    content=content.replace('with no nonfinite metrics.','with no nonfinite retained metrics after the disclosed exploratory correction.')
    marker='\\subsection{Information, memory, and dynamics}'
    content=content.replace(marker,marker+'\n\n\\paragraph{Numerical erratum.} '+DISCLOSURE)
    tex.write_text(content,encoding='utf-8')
    status=original.build_paper(paper,out/'report')
    original.require(all(x['success'] for x in status['builds'].values()),'Paper compilation failed')
    original.require(status['content_within_eight_pages'],'Paper body exceeds page limit')
    original.package_supplement(out)
    supplement=out/'report/supplement_anonymous.zip'
    with zipfile.ZipFile(supplement,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for f in (out/'correction_source_snapshot').rglob('*.py'):
            archive.write(f,f.relative_to(out).as_posix())
        for f in [out/'numerical_erratum_locked.json',out/'report/original_trace_identity.json']+list((out/'erratum_evidence').glob('*/seed_*.json')):
            archive.write(f,f.relative_to(out).as_posix())
        archive.writestr('NUMERICAL_ERRATUM.md',DISCLOSURE+'\n\nRun the original locked training/selection pipeline, then scripts_marl/run_paper_recovery.py with that original campaign as --origin and a fresh --out. The original failure and traces must be retained.\n')
    original.require(supplement.stat().st_size<=25*1024*1024,'Supplement exceeds 25 MB')
    write_json(out/'report/supplement_status.json',{'bytes':supplement.stat().st_size,'within_25mb':True,
        'sha256':sha(supplement),'numerical_erratum_included':True,'anonymity_manual_review_remaining':True})
    return result
