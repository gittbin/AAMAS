"""Descriptive, post-final winner-conditioned results; never changes primary evidence."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts_marl.run_paper_completion import ensure_gate, sha

OUT=ROOT/'outputs_marl/paper_completion_recovery'
REPORT=json.loads((OUT/'report/results.json').read_text(encoding='utf-8'))
AUDIT=json.loads((OUT/'report/audit_final.json').read_text(encoding='utf-8'))
assert REPORT['audit_passed'] and AUDIT['audit_passed']
ensure_gate(OUT,json.loads((OUT/'execution_protocol.json').read_text(encoding='utf-8')))
correction=json.loads((OUT/'numerical_erratum_locked.json').read_text(encoding='utf-8'))
assert sha(OUT/'numerical_erratum_locked.json')==REPORT['numerical_erratum']['correction_lock_sha256']
assert all(sha(ROOT/name)==digest for name,digest in correction['correction_source_files'].items())
SEEDS=REPORT['protocol']['training_seeds']
assert SEEDS==list(range(300,310))
EPS=1e-9
CACHE={}; RAW={}; GROUPS=[]; lines=[]
official={(r['condition']+'/'+r['treatment'],r['seed']):r for r in REPORT['per_seed']}
preserved=[OUT/'report/results.json',OUT/'report/audit_final.json',OUT/'selected_checkpoints_locked.json',
           OUT/'report/KET_QUA_NGAN_THEO_BANG_PAPER_DRAFT.md',OUT/'report/KET_QUA_TUNG_SEED_DIEN_PAPER.xlsx']
before={p:sha(p) for p in preserved}


def vector(key,metric='net_return'):
    if (key,metric) not in CACHE:
        if key.startswith('baseline/'):
            result=np.full(10,REPORT['summaries'][key]['metrics'][metric]['mean'])
        elif metric in official[(key,SEEDS[0])]:
            result=np.array([official[(key,seed)][metric] for seed in SEEDS],dtype=float)
        else:
            condition,treatment=key.split('/')
            numbers=[]
            for seed in SEEDS:
                if (condition,seed) not in RAW:
                    RAW[(condition,seed)]=json.loads((OUT/f'evaluation/{condition}/seed_{seed}/results.json').read_text(encoding='utf-8'))
                episodes=RAW[(condition,seed)]['treatments'][treatment]['episodes']
                numbers.append(float(np.mean([e[metric] for e in episodes])))
            result=np.asarray(numbers)
        assert result.shape==(10,) and np.isfinite(result).all()
        CACHE[(key,metric)]=result
    return CACHE[(key,metric)]


def pair(name,left,right,metric='net_return'):
    return name,left,right,metric


def chosen(spec):
    _,left,right,metric=spec
    return vector(left,metric)-vector(right,metric)>EPS


def labels(mask):
    return ', '.join(f'S{i}' for i in np.flatnonzero(mask)) or 'Không có'


def number(value,signed=False):
    if value is None: return '—'
    return f'{value:+.4f}' if signed else f'{value:.3f}'


def average(key,metric,mask):
    return float(vector(key,metric)[mask].mean()) if mask.any() else None


def table(headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
    lines.append('')


def comparisons(specs):
    rows=[]
    for spec in specs:
        name,left,right,metric=spec; mask=chosen(spec)
        difference=vector(left,metric)-vector(right,metric)
        n=int(mask.sum())
        assert n==sum(d>EPS for d in difference)
        selected=float(difference[mask].mean()) if n else None
        if n:
            assert selected>EPS and np.isclose(selected,average(left,metric,mask)-average(right,metric,mask),atol=1e-10,rtol=1e-10)
        rows.append([name,f'{n}/10',labels(mask),number(average(left,metric,mask)),number(average(right,metric,mask)),
                     number(selected,True),number(float(difference.mean()),True)])
        GROUPS.append({'comparison':name,'metric':metric,'n':n,'seed_indices':np.flatnonzero(mask).tolist(),
            'mean_all':float(difference.mean()),'mean_winners':selected})
    table(['So sánh (trái − phải)','Số thắng','Giữ seed','Trái: mean nhóm thắng','Phải: cùng nhóm seed','Δ nhóm thắng','Δ đủ 10 seed'],rows)


def matched_means(entries,mask,metrics=('net_return','min_user_rate','illumination_db'),headings=('J','R','B')):
    if not mask.any():
        lines.append('Không có seed thỏa điều kiện; không có trung bình nhóm thắng để tính.')
        return
    table(['Dòng/model']+list(headings),[[name]+[number(average(key,m,mask)) for m in metrics] for name,key in entries])


def raw_winners(specs):
    rows=[]
    for spec in specs:
        name,left,right,metric=spec
        for i in np.flatnonzero(chosen(spec)):
            a,b=vector(left,metric)[i],vector(right,metric)[i]
            rows.append([name,f'S{i}',number(a),number(b),number(a-b,True)])
    if rows: table(['So sánh','Seed thắng','Trái','Phải','Δ'],rows)


primary_names={'adaptation':('Table 3','Adaptive − Frozen25, R post25'),
    'full_vs_ma':('Table 4','Full − MA, R'),'full_vs_power_ris':('Table 4','Full − Power+RIS, R'),
    'full_vs_power_ma':('Table 4','Full − Power+MA, R'),'full_vs_ris_ma':('Table 4','Full − RIS+MA, R'),
    'comm_vs_nocomm':('Table 5','Comm − NoComm, J'),'communication_on_vs_off':('Table 5','Normal − Off, J')}
lines += ['# Kết quả riêng các seed thắng — theo paper_draft.pdf','',
    '**Phân tích mô tả sau khi xem final (post hoc).** Chọn seed thắng làm mean thuận lợi hơn theo chính điều kiện chọn; các số này không thay thế kết quả đủ 10 seed trong paper và không xác nhận significance mới.',
    '**S0…S9 = seed 300…309.** J = return; R = min-user rate (bit/s/Hz); B = illumination dB. Thắng khi trái − phải > 1e-9 trên số chưa làm tròn; hòa được loại khỏi nhóm thắng.',
    '**Mỗi so sánh có nhóm seed thắng riêng.** Cả hai model luôn lấy đúng cùng nhóm seed, giữ đủ 150 scenario/seed (30 cho robustness). Không lọc riêng episode. Baseline không train vẫn dùng 150 scenario chung, không tạo training seed giả.',
    '**Cách đọc:** “Δ nhóm thắng” là mean chỉ trong nhóm thắng; “Δ đủ 10 seed” là kết quả gốc. Dấu — nghĩa là không có seed thắng, không phải số 0. Không tính lại CI/p-value trên nhóm được chọn theo kết quả.']
rows=[]
for h in REPORT['primary']:
    spec=pair(primary_names[h['name']][1],h['left'],h['right'],h['metric']); mask=chosen(spec)
    difference=vector(h['left'],h['metric'])-vector(h['right'],h['metric'])
    assert int(mask.sum())==h['wins'] and np.isclose(difference.mean(),h['mean'],atol=1e-10,rtol=1e-10)
    rows.append([primary_names[h['name']][0],spec[0],f'{int(mask.sum())}/10',labels(mask),
                 number(float(difference[mask].mean()) if mask.any() else None,True),number(h['mean'],True),f'{h["p_holm"]:.4f}'])
table(['Bảng','So sánh','Số thắng','Seed giữ lại','Δ chỉ seed thắng','Δ đủ 10 seed','p Holm gốc: 10 seed'],rows)

lines += ['','## Table 3 — Controller/adaptation, mục 6.1']
adapt=pair('Adaptive − Frozen25, R post25','full_nocomm/normal','full_nocomm/frozen25','post25_min_user_rate')
mask=chosen(adapt)
lines += [f'**Nhóm thắng adaptation: {labels(mask)} ({int(mask.sum())}/10); loại {labels(~mask)}.** Các mean dưới đây cùng dùng nhóm này, kể cả Comm/Central; không chọn lại seed tốt nhất cho từng dòng.']
controllers=[('NoComm adaptive','full_nocomm/normal'),('Frozen25','full_nocomm/frozen25'),('Static learned','full_nocomm/static_learned'),
    ('Comm','full_comm/normal'),('Central','centralized/normal'),('USPA no-op','baseline/noop'),('Archive knee','baseline/ramp_knee'),
    ('Archive rate extreme','baseline/ramp_comm_extreme'),('Archive sensing extreme','baseline/ramp_sensing_extreme'),
    ('Heuristic','baseline/heuristic'),('Random','baseline/random'),('Online optimizer','baseline/online_optimizer')]
matched_means(controllers,mask,('net_return','min_user_rate','sum_rate','illumination_db','outage_fraction','delivered'),
              ('J','R','Sum-rate','B','Outage','Delivered'))
raw_winners([adapt])
lines += ['**Nếu định nghĩa thắng controller theo J**, nhóm seed thay đổi:']
comparisons([pair('NoComm − Central, J','full_nocomm/normal','centralized/normal'),
    pair('NoComm − static learned, J','full_nocomm/normal','full_nocomm/static_learned'),
    pair('NoComm − archive knee, J','full_nocomm/normal','baseline/ramp_knee'),
    pair('NoComm − online, J','full_nocomm/normal','baseline/online_optimizer')])

lines += ['','## Table 4 — Actuator subsets, mục 6.2']
subsets=[('Power','power'),('RIS','ris'),('MA','ma'),('Power+RIS','power_ris'),('Power+MA','power_ma'),('RIS+MA','ris_ma')]
specs=[pair('Full − '+name+', R','full_nocomm/normal',key+'/normal','min_user_rate') for name,key in subsets]
comparisons(specs)
pm=next(s for s in specs if s[2]=='power_ma/normal')
lines += ['**Chi tiết nhóm Full thắng Power+MA:**']
raw_winners([pm])
joint=np.logical_and.reduce([chosen(s) for s in specs])
lines += [f'**Một nhóm chung để lập lại nguyên Table 4:** Full thắng cả 6 subsets theo R: **{labels(joint)} ({int(joint.sum())}/10)**.']
matched_means([(name,key+'/normal') for name,key in subsets]+[('Full','full_nocomm/normal')],joint)
lines += ['Thắng R không mặc nhiên là thắng J hoặc B. Nhóm 4 seed thắng Power+MA và nhóm thắng tất cả subsets là hai nhóm khác nhau.']

lines += ['','## Mục 6.3 — Memory/CSI/dynamics']
memory=[pair('Rec − FF, J','full_nocomm/normal','feedforward/normal'),
    pair('Rec − Masked, J','full_nocomm/normal','masked_csi/normal'),
    pair('Rec − Fixed CSI, J','full_nocomm/normal','full_nocomm/csi_fixed'),
    pair('Rec − Shuffled CSI, J','full_nocomm/normal','full_nocomm/csi_shuffled')]
comparisons(memory)
comparisons([pair('Adaptive − Frozen25: '+v+', R post25','full_nocomm/dynamics_'+v,
    'full_nocomm/dynamics_'+v+'_frozen25','post25_min_user_rate') for v in ('static','slow','fast')])
lines += ['**Đường cong: chỉ lấy seed Rec thắng FF về R ở từng điểm**; đây không còn là đường trung bình của toàn bộ 10 seed.']
curves=[]
for kind,grid in [('tau',REPORT['protocol']['tau_grid']),('delay',REPORT['protocol']['delay_grid'])]:
    for value in grid:
        curves.append(pair(f'{kind}={value}: Rec − FF, R','full_nocomm/'+f'{kind}_{value}',
                           'feedforward/'+f'{kind}_{value}','min_user_rate'))
comparisons(curves)
lines += ['Giữ nguyên numerical erratum đã công khai cho τ=0; không thay model, checkpoint hay scenario.']

lines += ['','## Table 5 — Communication, mục 6.4']
comm=pair('Comm − NoComm, J','full_comm/normal','full_nocomm/normal')
onoff=pair('Normal − Off, J','full_comm/normal','full_comm/comm_off')
comparisons([comm,onoff]); raw_winners([comm,onoff])
joint_comm=chosen(comm)&chosen(onoff)
lines += [f'**Thắng đồng thời Comm>NoComm và Normal>Off: {labels(joint_comm)} ({int(joint_comm.sum())}/10).** Không ghép hai nhóm khác nhau thành một nhóm chứng minh communication có ích.']
payloads=[pair('Normal − '+label+', J','full_comm/normal','full_comm/'+treatment) for label,treatment in (
    ('Zero','content_zero'),('Random payload','content_random'),('Permuted payload','content_permute'),
    ('Random delivery','comm_random'),('Forced delivery','comm_forced'))]
comparisons(payloads)
lines += ['**Harmful graph 52: chỉ giữ seed Normal harmful thắng payload bị sửa.**']
comparisons([pair('Harmful normal − '+v+', J','full_comm/graph_harmful','full_comm/graph_harmful_content_'+v)
    for v in ('zero','random','permute')])
control=json.loads((ROOT/'outputs_marl/paper_v2/diagnostics/results.json').read_text(encoding='utf-8'))
toy=[]
for price in (0.1,1.0):
    runs=[r for r in control['runs'] if r['task']=='cost_only' and r['price']==price
          and r['after']['requests_per_step']<0.2*r['before']['requests_per_step']]
    toy.append([f'Cost-only price {price:g}','Requests sau <20% trước',', '.join(str(r['training_seed']) for r in runs),
                number(float(np.mean([r['after']['requests_per_step'] for r in runs])) if runs else None)])
runs=[r for r in control['runs'] if r['task']=='positive' and r['after']['accuracy']>=0.8]
toy.append(['Private bit','Accuracy ≥80%',', '.join(str(r['training_seed']) for r in runs),
    f'{100*np.mean([r["after"]["accuracy"] for r in runs]):.2f}%' if runs else '—'])
lines += ['**Capability controls:** toy seed 0–2 riêng; lọc theo tiêu chí diagnostic đã dùng, không theo wireless return.']
table(['Control','Tiêu chí đạt','Toy seed giữ','Mean sau train'],toy)

lines += ['','## Table 6 — Price/budget, mục 6.4']
prices=[pair('B1 free − default, J','price_free/normal','full_comm/normal'),
    pair('B1 tenfold − default, J','price_tenfold/normal','full_comm/normal'),
    pair('B3 free − default, J','budget3_free/normal','budget3_default/normal'),
    pair('B3 tenfold − default, J','budget3_tenfold/normal','budget3_default/normal'),
    pair('Requested − B1 default, J','request_price/normal','full_comm/normal'),
    pair('B3 default − B1 default, J','budget3_default/normal','full_comm/normal')]
comparisons(prices)
lines += ['Mỗi dòng giữ giá thực tế của condition. Nhóm seed khác nhau theo từng cặp; B3 còn đổi kích thước inbox.']

lines += ['','## Table 7 — Graphs, mục 6.5']
graphs=[('Empty=Selected (0)','empty'),('Ring (25)','ring'),('BS star (23)','bs_star'),('Harmful (52)','harmful')]
graph_specs=[pair(name+' − Full graph, R','full_comm/graph_'+g,'full_comm/graph_full','min_user_rate') for name,g in graphs]
comparisons(graph_specs+[pair('Retrained graph0 − NoComm, R','selected_graph/normal','full_nocomm/normal','min_user_rate')])
graph_group=chosen(graph_specs[0])
lines += [f'**Lập lại Table 7 trên một nhóm chung:** dùng seed Empty/Selected thắng Full graph: **{labels(graph_group)} ({int(graph_group.sum())}/10)**.']
matched_means([(name,'full_comm/graph_'+g) for name,g in (
    ('Empty/Selected (0)','empty'),('Full (63)','full'),('Ring (25)','ring'),('BS star (23)','bs_star'),('Harmful (52)','harmful'))],
    graph_group,('min_user_rate','illumination_db','delivered'),('R','B','Delivered'))
retrain_diff=vector('selected_graph/normal','min_user_rate')-vector('full_nocomm/normal','min_user_rate')
lines += [f'Retrained graph0 so với NoComm: **{int((np.abs(retrain_diff)<=EPS).sum())}/10 seed hòa theo R**; không coi hòa là thắng.']

lines += ['','## Table 8 — Latency, mục 6.6']
timing=json.loads((OUT/'baselines/latency.json').read_text(encoding='utf-8'))
table(['Model','Policy seed đã đo','Decision kernels median ms','P95 ms'],[
    [label,'S0 (300)' if key!='online_optimizer' else 'Không train',
     number(timing[key]['summary']['decision_kernels_ms']['median']),number(timing[key]['summary']['decision_kernels_ms']['p95'])]
    for label,key in [('NoComm','full_nocomm'),('Comm','full_comm'),('Central','centralized'),('Online','online_optimizer')]])
lines += ['Benchmark chỉ có S0, 300 measurement sau 30 warm-up. Giữ số đo gốc; không suy ra latency cho các nhóm seed thắng khác và không chọn measurement nhanh nhất. Đây là decision kernels, chưa phải end-to-end.',
    '', '**Cách dùng:** xem nhóm nào hoạt động tốt và độ chênh trong nhóm đó. Báo cáo chính của paper vẫn giữ đầy đủ 10 seed và p/CI gốc; nhóm được chọn sau khi xem kết quả cần ghi rõ là phân tích hậu nghiệm.',
    '**Nguồn:** cùng dữ liệu đã full audit với file ngắn trước. Các report, checkpoint, protocol, file Markdown trước và Excel được giữ nguyên.']

assert all(sha(path)==digest for path,digest in before.items())
destination=OUT/'report/KET_QUA_CHI_SEED_THANG_THEO_PAPER_DRAFT.md'
destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'path':str(destination),'lines':len(lines),'bytes':destination.stat().st_size,
    'descriptive_post_hoc':True,'new_p_values_or_inferential_ci':False,'original_artifacts_unchanged':True,
    'pair_groups':len(GROUPS),'joint_communication_winners':np.flatnonzero(joint_comm).tolist(),
    'joint_all_actuator_winners':np.flatnonzero(joint).tolist()},indent=2))
