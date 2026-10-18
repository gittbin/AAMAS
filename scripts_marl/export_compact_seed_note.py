"""Compact, table-numbered seed results for the archived original paper draft."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts_marl.run_paper_completion import ensure_gate

OUT=ROOT/'outputs_marl/paper_completion_recovery'
REPORT=json.loads((OUT/'report/results.json').read_text(encoding='utf-8'))
AUDIT=json.loads((OUT/'report/audit_final.json').read_text(encoding='utf-8'))
assert AUDIT['audit_passed'] and REPORT['audit_passed']
ensure_gate(OUT,json.loads((OUT/'execution_protocol.json').read_text(encoding='utf-8')))
SEEDS=REPORT['protocol']['training_seeds']
assert SEEDS==list(range(300,310))
CACHE={}
lines=[]


def values(key,metric='net_return'):
    cache_key=(key,metric)
    if cache_key not in CACHE:
        condition,treatment=key.split('/')
        CACHE[cache_key]=[]
        for seed in SEEDS:
            data=json.loads((OUT/f'evaluation/{condition}/seed_{seed}/results.json').read_text(encoding='utf-8'))
            eps=data['treatments'][treatment]['episodes']
            number=float(np.mean([e[metric] for e in eps]))
            if metric in ('net_return','min_user_rate','sum_rate','illumination_db','outage_fraction','requested','delivered'):
                official=next(r for r in REPORT['per_seed'] if r['condition']==condition and r['treatment']==treatment and r['seed']==seed)
                assert np.isclose(number,official[metric],rtol=1e-10,atol=1e-10)
            CACHE[cache_key].append(number)
        CACHE[cache_key]=np.asarray(CACHE[cache_key])
    return CACHE[cache_key]


def mean(key,metric='net_return'):
    return REPORT['summaries'][key]['metrics'][metric]['mean']


def mark(difference):
    return '✓' if difference>1e-9 else '×' if difference< -1e-9 else '='


def cell(value,difference=None):
    return f'{value:.3f}'+(' '+mark(difference) if difference is not None else '')


def wins(left,right,metric='net_return'):
    differences=values(left,metric)-values(right,metric)
    groups=[]
    for symbol in ('✓','×','='):
        selected=[f'S{i}' for i,d in enumerate(differences) if mark(d)==symbol]
        if selected: groups.append(symbol+' '+', '.join(selected))
    return '; '.join(groups)


def table(headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    for row in rows: lines.append('| '+' | '.join(map(str,row))+' |')
    lines.append('')


def aggregate(entries,metrics,labels):
    table(['Dòng trong paper']+labels,[[name]+[cell(mean(key,m)) for m in metrics] for name,key in entries])


lines += ['# Kết quả ngắn theo các bảng trong paper_draft.pdf','',
    'Số bảng theo **paper_draft.pdf, trang 8–9, Table 3–8**. Table 1–2 là thông số, không có thắng/thua.',
    '**S0…S9 = training seed 300…309.** Mỗi seed được đánh giá trên cùng 150 scenario; đường cong τ/delay/dynamics dùng 30 scenario. Các bảng wireless dùng nghiên cứu final mới; capability controls toy được ghi riêng ở Table 5.',
    '**✓ = thắng đối chứng được nêu, × = thua, = = hòa**, tính trên số gốc của cùng training seed. Đây là thắng về số đo; ý nghĩa thống kê xem bảng kiểm định cuối file. Số hiển thị làm tròn 3 chữ số.',
    '**J** = return/episode; **R** = min-user rate (bit/s/Hz); **B** = illumination dB. Các bảng tổng hợp là mean. Full = Power+RIS+MA, NoComm = Full học không communication, Comm = Full học có communication.']

lines += ['','## Table 3 — Controller comparison / mục 6.1']
entries=[('USPA no-op','baseline/noop'),('Archive knee','baseline/ramp_knee'),
    ('Archive communication extreme','baseline/ramp_comm_extreme'),('Archive sensing extreme','baseline/ramp_sensing_extreme'),
    ('Estimated-channel heuristic','baseline/heuristic'),('Random physical control','baseline/random'),
    ('Policy-selected frozen (freeze25)','full_nocomm/frozen25'),('MAPPO without communication','full_nocomm/normal'),
    ('MAPPO with communication','full_comm/normal'),('Centralized recurrent PPO','centralized/normal'),
    ('Online optimizer','baseline/online_optimizer'),('Bổ sung: static learned trên development','full_nocomm/static_learned')]
aggregate(entries,('net_return','min_user_rate','sum_rate','illumination_db','outage_fraction','delivered'),['J','R','Sum-rate','B','Outage','Messages'])
lines += ['**Từng seed: J của các controller có training seed.** Cột cuối là ΔR sau bước 25: Adaptive − Frozen25. Static learned là một đối chứng riêng, được chọn trên development.']
keys=['full_nocomm/normal','full_comm/normal','centralized/normal','full_nocomm/frozen25','full_nocomm/static_learned']
adaptive=values('full_nocomm/normal','post25_min_user_rate')-values('full_nocomm/frozen25','post25_min_user_rate')
table(['Seed','NoComm J','Comm J','Central J','Frozen25 J','Static learned J','ΔR post25'],
    [[f'S{i}']+[cell(values(k)[i]) for k in keys]+[f'{adaptive[i]:+.3f} {mark(adaptive[i])}'] for i in range(10)])
lines += ['**Adaptation thắng/thua:** '+wins('full_nocomm/normal','full_nocomm/frozen25','post25_min_user_rate')+'.',
    '**Chốt Table 3:** có bằng chứng tiếp tục thích nghi sau freeze25. Mean J của Archive knee cao hơn NoComm; không viết NoComm thắng mọi static baseline.']

lines += ['','## Table 4 — Actuator subsets / mục 6.2']
subsets=[('Power','power'),('RIS','ris'),('MA','ma'),('Power+RIS','power_ris'),('Power+MA','power_ma'),('RIS+MA','ris_ma'),('Full','full_nocomm')]
aggregate([(name,key+'/normal') for name,key in subsets],('net_return','min_user_rate','illumination_db'),['J','R','B'])
lines += ['**Từng seed: R.** Dấu cạnh mỗi subset cho biết **Full thắng/thua subset đó**; cột Full là R của Full.']
full=values('full_nocomm/normal','min_user_rate')
table(['Seed']+[name for name,_ in subsets],[[f'S{i}']+[
    cell(values(key+'/normal','min_user_rate')[i],full[i]-values(key+'/normal','min_user_rate')[i]) if key!='full_nocomm'
    else cell(full[i]) for _,key in subsets] for i in range(10)])
lines += ['**Full so với Power+MA:** '+wins('full_nocomm/normal','power_ma/normal','min_user_rate')+'.',
    '**Chốt Table 4:** Full hơn MA, Power+RIS và RIS+MA; chưa xác nhận hơn Power+MA. Không kết luận cả ba actuator đều cần thiết.']

lines += ['','## Mục 6.3 — Memory, CSI và dynamics (draft không có bảng đánh số)']
memory=[('Rec','full_nocomm/normal'),('FF','feedforward/normal'),('Masked','masked_csi/normal'),
        ('Fixed CSI','full_nocomm/csi_fixed'),('Shuffled CSI','full_nocomm/csi_shuffled')]
lines += ['**Từng seed: J.** Dấu ở FF/Masked cho biết Rec thắng/thua đối chứng; Fixed/Shuffled là can thiệp lên Rec.']
table(['Seed']+[n for n,_ in memory],[[f'S{i}']+[cell(values(k)[i],values('full_nocomm/normal')[i]-values(k)[i])
    if n in ('FF','Masked') else cell(values(k)[i]) for n,k in memory] for i in range(10)])
lines += ['**Dynamics: ΔR post25 = Adaptive − Frozen25**, từng seed, chỉ là exploratory.']
table(['Seed','Static','Slow','Fast'],[[f'S{i}']+[cell(
    values('full_nocomm/dynamics_'+v,'post25_min_user_rate')[i]-values('full_nocomm/dynamics_'+v+'_frozen25','post25_min_user_rate')[i],
    values('full_nocomm/dynamics_'+v,'post25_min_user_rate')[i]-values('full_nocomm/dynamics_'+v+'_frozen25','post25_min_user_rate')[i]) for v in ('static','slow','fast')] for i in range(10)])
curve_rows=[]
for kind,grid in [('tau',REPORT['protocol']['tau_grid']),('delay',REPORT['protocol']['delay_grid'])]:
    for v in grid:
        keys=[c+f'/{kind}_{v}' for c in ('full_nocomm','feedforward','masked_csi')]
        curve_rows.append([f'{kind}={v}']+[cell(mean(k,'min_user_rate')) for k in keys]+[wins(keys[0],keys[1],'min_user_rate')])
table(['Điểm đường cong','Rec R','FF R','Masked R','Rec so với FF: seed thắng/thua'],curve_rows)
lines += ['**Chốt mục 6.3:** chưa xác nhận ưu thế recurrent hoặc CSI trên nominal return. Không dùng đường delay phẳng để kết luận memory bù được delay. Điểm τ=0 dùng numerical erratum được công khai; giữ nguyên mọi checkpoint/seed.']

lines += ['','## Table 5 — Communication matrix / mục 6.4']
communication=[('NoComm','full_nocomm/normal'),('Normal','full_comm/normal'),('Off','full_comm/comm_off'),
    ('Zero','full_comm/content_zero'),('Random payload','full_comm/content_random'),('Permuted payload','full_comm/content_permute'),
    ('Random delivery','full_comm/comm_random'),('Forced delivery','full_comm/comm_forced')]
aggregate(communication,('net_return','min_user_rate','requested','delivered'),['J','R','Requests','Delivered'])
lines += ['**Từng seed: J.** Dấu cạnh đối chứng cho biết **Normal thắng/thua đối chứng**; NoComm là mô hình học riêng, các cột khác dùng checkpoint Comm cố định.']
table(['Seed']+[n for n,_ in communication],[[f'S{i}']+[
    cell(values(k)[i],values('full_comm/normal')[i]-values(k)[i]) if n!='Normal' else cell(values(k)[i])
    for n,k in communication] for i in range(10)])
lines += ['**Comm − NoComm:** '+wins('full_comm/normal','full_nocomm/normal')+'.',
    '**Normal − Off:** '+wins('full_comm/normal','full_comm/comm_off')+'.',
    '**Lặp trên harmful graph ID 52: J từng seed.** Dấu = Normal harmful thắng/thua payload bị sửa; send/delivery được giữ cố định.']
harmful=[('Normal harmful','full_comm/graph_harmful')]+[(n,'full_comm/graph_harmful_content_'+k) for n,k in [('Zero','zero'),('Random','random'),('Permuted','permute')]]
table(['Seed']+[n for n,_ in harmful],[[f'S{i}']+[
    cell(values(k)[i],values(harmful[0][1])[i]-values(k)[i]) if n!='Normal harmful' else cell(values(k)[i]) for n,k in harmful] for i in range(10)])
control=json.loads((ROOT/'outputs_marl/paper_v2/diagnostics/results.json').read_text(encoding='utf-8'))
toy=[]
for seed in (0,1,2):
    runs=[r for r in control['runs'] if r['training_seed']==seed]
    costs=[next(r for r in runs if r['task']=='cost_only' and r['price']==p) for p in (0,0.1,1)]
    bit=next(r for r in runs if r['task']=='positive')
    toy.append([seed]+[f'{r["before"]["requests_per_step"]:.3f} → {r["after"]["requests_per_step"]:.3f}' for r in costs]+[f'{100*bit["after"]["accuracy"]:.2f}%'])
lines += ['**Capability controls của draft:** toy seed 0–2, 16.384 transitions; không gộp với S0–S9.']
table(['Toy seed','Requests free: trước→sau','Requests giá 0.1','Requests giá 1','Private-bit accuracy'],toy)
lines += ['**Chốt Table 5:** communication chưa có lợi ích net return được xác nhận. Normal − random payload có CI exploratory dương; trên harmful graph, zero payload lại cải thiện return. Không kết luận payload vô dụng ở mọi graph.']

lines += ['','## Table 6 — Price conditions / mục 6.4']
prices=[('B1 free','price_free'),('B1 default','full_comm'),('B1 tenfold','price_tenfold'),
        ('B3 free','budget3_free'),('B3 default','budget3_default'),('B3 tenfold','budget3_tenfold'),('Requested price','request_price')]
aggregate([(n,k+'/normal') for n,k in prices],('net_return','min_user_rate','requested','delivered'),['J','R','Requests','Delivered'])
lines += ['**Từng seed: J.** Dấu so với default cùng budget; Requested price so với B1 default. Return giữ đúng giá của từng condition.']
table(['Seed']+[n for n,_ in prices],[[f'S{i}']+[
    cell(values(k+'/normal')[i],values(k+'/normal')[i]-values(('budget3_default' if k.startswith('budget3_') else 'full_comm')+'/normal')[i])
    if k not in ('full_comm','budget3_default') else cell(values(k+'/normal')[i]) for _,k in prices] for i in range(10)])
lines += ['**Chốt Table 6:** mô tả trade-off giá và physical metrics, chưa có primary test về price. B3 còn tăng kích thước inbox/actor input, nên không gọi đây là can thiệp bandwidth thuần túy.']

lines += ['','## Table 7 — Graph evaluation / mục 6.5']
graphs=[('Empty (0)','empty'),('Full (63)','full'),('Ring (25)','ring'),('BS star (23)','bs_star'),('Selected Pareto (0)','selected'),('Harmful (52)','harmful')]
aggregate([(n,'full_comm/graph_'+k) for n,k in graphs],('min_user_rate','illumination_db','delivered'),['R','B','Delivered'])
lines += ['**Từng seed: R.** Dấu cho biết graph đang xét thắng/thua Full graph. Empty = Selected, nên gộp một cột. Retrain là policy được train mới với graph 0.']
graph_seed=[('Empty=Selected','full_comm/graph_empty'),('Full','full_comm/graph_full'),('Ring','full_comm/graph_ring'),
            ('BS star','full_comm/graph_bs_star'),('Harmful','full_comm/graph_harmful'),('Retrain','selected_graph/normal')]
table(['Seed']+[n for n,_ in graph_seed],[[f'S{i}']+[
    cell(values(k,'min_user_rate')[i],values(k,'min_user_rate')[i]-values('full_comm/graph_full','min_user_rate')[i])
    if n not in ('Full','Retrain') else cell(values(k,'min_user_rate')[i]) for n,k in graph_seed] for i in range(10)])
g=json.loads((OUT/'graph_selection.json').read_text(encoding='utf-8'))
lines += [f'**Chốt Table 7:** development Pareto front có {len(g["pareto_front"])} graph; selected ID 0 (không cạnh). Retrain: J={mean("selected_graph/normal"):.3f}, R={mean("selected_graph/normal","min_user_rate"):.3f}. Kết quả conditional theo frozen policies; không phải optimum toàn cục graph–policy.']

lines += ['','## Table 8 — Latency / mục 6.6']
timing=json.loads((OUT/'baselines/latency.json').read_text(encoding='utf-8'))
latency=[('Decentralized inference — NoComm','full_nocomm','actor_or_optimizer_ms'),
    ('Router/shield — NoComm','full_nocomm','routing_shield_ms'),
    ('Centralized inference','centralized','actor_or_optimizer_ms'),('Online optimization','online_optimizer','actor_or_optimizer_ms'),
    ('Common BD/power — NoComm','full_nocomm','low_level_bd_power_ms'),
    ('Decentralized decision kernels — NoComm','full_nocomm','decision_kernels_ms'),
    ('Decentralized decision kernels — Comm','full_comm','decision_kernels_ms'),
    ('Centralized decision kernels','centralized','decision_kernels_ms'),
    ('Online decision kernels','online_optimizer','decision_kernels_ms')]
table(['Dòng điền Table 8','Median ms','P95 ms'],[[n,cell(timing[c]['summary'][m]['median']),cell(timing[c]['summary'][m]['p95'])] for n,c,m in latency])
speed=timing['online_optimizer']['summary']['decision_kernels_ms']['median']/timing['full_nocomm']['summary']['decision_kernels_ms']['median']
lines += [f'Chỉ đo policy **S0 (seed 300)** cho các learned controller, 30 warm-up + 300 lần đo, CPU một Torch thread, batch 1. Không có latency 10 training seed. Median online/NoComm ≈ **{speed:.2f}×**.',
    '**Sửa hai nhãn “end-to-end” của draft thành “decision kernels”.** Chưa đo acquisition, dựng channel/features, dynamics hay transport. Các mục end-to-end nguyên nghĩa chưa có số để điền.',
    f'NoComm: gap J tới online **{REPORT["performance_relative_to_online"]["full_nocomm"]["absolute_gap_to_online"]:+.3f}**, raw ratio **{REPORT["performance_relative_to_online"]["full_nocomm"]["raw_return_ratio_to_online"]:.3f}**. Gain recovery không định nghĩa vì online kém static learned.',
    'Audit đạt: 180 lượt train, 72.069.120 transitions, 69.750 episode / 6.975.000 trace rows; 0 NaN/Inf trong dữ liệu kết quả giữ lại. Numerical erratum sau khi mở final được ghi rõ trong audit.']

lines += ['','## Chốt các kiểm định chính để viết kết luận']
names={'adaptation':('Table 3','Adaptive − Frozen25, R post25'),'full_vs_ma':('Table 4','Full − MA, R'),
       'full_vs_power_ris':('Table 4','Full − P+RIS, R'),'full_vs_power_ma':('Table 4','Full − P+MA, R'),
       'full_vs_ris_ma':('Table 4','Full − RIS+MA, R'),'comm_vs_nocomm':('Table 5','Comm − NoComm, J'),
       'communication_on_vs_off':('Table 5','Normal − Off, J')}
table(['Bảng','So sánh','Δ mean','CI95','p Holm','Seed thắng','Kết luận'],[[*names[h['name']],f'{h["mean"]:+.3f}',
    f'[{h["ci95"][0]:+.3f}; {h["ci95"][1]:+.3f}]',f'{h["p_holm"]:.4f}',f'{h["wins"]}/10',
    'Đạt' if h['p_holm']<0.05 and h['mean']>0 else 'Chưa xác nhận thắng'] for h in REPORT['primary']])
lines += ['CI là pointwise nested bootstrap; p Holm chỉnh cả 7 primary tests. Nonsignificance không chứng minh tương đương.',
    '**Câu kết luận phù hợp:** Adaptation sau warm-up có lợi; Full hơn một số subsets nhưng chưa hơn Power+MA; communication chưa có lợi ích net return được xác nhận trong thiết lập này.',
    '', '**Nguồn:** `report/results.json`, các evaluation results và `baselines/latency.json`, đã full audit. Đây là bản tóm tắt theo draft; file Excel cũ được giữ nguyên.']

destination=OUT/'report/KET_QUA_NGAN_THEO_BANG_PAPER_DRAFT.md'
for h in REPORT['primary']:
    difference=values(h['left'],h['metric'])-values(h['right'],h['metric'])
    assert np.isclose(difference.mean(),h['mean'],rtol=1e-10,atol=1e-10)
    assert sum(mark(d)=='✓' for d in difference)==h['wins']
destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'path':str(destination),'lines':len(lines),'bytes':destination.stat().st_size,
                  'seed_values_checked_against_audited_report':True},indent=2))
