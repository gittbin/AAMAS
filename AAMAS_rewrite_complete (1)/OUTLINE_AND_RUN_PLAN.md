# Outline và kế hoạch hoàn thiện paper AAMAS

Bản viết lại: **Heterogeneous Multi-Agent Coordination in Programmable 6G Networks under Delayed Partial Observability and Costly Communication**.

Đây là manuscript nghiên cứu đầy đủ để triển khai thí nghiệm. Không có số liệu giả, p-value giả, hay kết luận rằng thí nghiệm mới đã thành công. Các kết quả cũ không được tự động chuyển sang evaluator đã sửa.

## Câu chuyện xuyên suốt

Ba điều cần chứng minh riêng:

1. **Adaptation:** policy tiếp tục điều chỉnh có tốt hơn cấu hình do chính policy chọn rồi giữ nguyên không?
2. **Coordination:** team nhiều actuator có thêm giá trị so với MA-only và các actuator pairs không?
3. **Communication:** gửi thông tin có cải thiện quyết định đủ để bù cost, delay và budget không?

Không cần ép communication phải có lợi. Một negative result có đối chứng cơ chế tốt vẫn phù hợp với câu hỏi của bài. Nếu communication trở nên có ích sau thay đổi lớn về thiết kế, cần đối chiếu abstract đã đăng ký và hỏi chairs về cập nhật nội dung đáng kể.

## Cấu trúc manuscript

| Phần | Nội dung được viết | Bằng chứng cần điền sau |
|---|---|---|
| Abstract | Bài toán, CTDE, delay/cost/budget và cách tách các hiệu ứng | Tóm tắt kết quả đã kiểm chứng |
| 1. Introduction | Motivation multi-agent; bốn RQ; contributions | Contribution thực nghiệm cuối cùng |
| 2. Related Work | CTDE/MAPPO, communication, programmable wireless, đánh giá RL | Bổ sung literature sát nhất trước khi nộp |
| 3. System and Information Model | Geometry, channels, estimated/true CSI, residual interference, sensing, dynamics | Kiểm tra phương trình với evaluator |
| 4. Dec-POMDP and Controller | State/history/queue, observations, codebooks, scheduler/shield, router, reward, PPO/GAE, graph enumeration | Schema/config thực sự dùng |
| 5. Experimental Protocol | Baselines; adaptation/memory/actuator/message controls; statistics; latency | Khóa config trước final runs |
| 6. Results | Sáu nhóm kết quả; các bảng trống có sẵn | Mean, CI, tests và đồ thị thật |
| 7. Discussion and Limitations | Các giới hạn và cách diễn giải theo bằng chứng | Kết nối kết quả với RQ |
| 8. Conclusion | Tóm tắt framework và thiết kế nghiên cứu | Kết luận định lượng cuối cùng |

## Hình đã tạo

- `figures/system_model.*`: hình vật lý MA–RIS–ISAC, direct/reflected communication và sensing paths; minh họa không theo tỷ lệ.
- `figures/ctde_architecture.*`: luồng CSI/local observation, recurrent actors, router, shield, low-level precoding và critic training-only.
- Mỗi hình có PDF vector, SVG chỉnh sửa được và PNG xem nhanh. Số biểu tượng người dùng/RIS chỉ để minh họa; không biểu diễn kích thước experiment.

## P0: kiểm tra trước khi train lại

- [ ] Đối chiếu SINR trong manuscript với code: residual inter-user interference xuất hiện khi BD dùng estimated CSI.
- [ ] Đối chiếu EVM convention: signal scaling, distortion covariance, tổng transmit power.
- [ ] Kiểm tra nulling ở perfect CSI bằng residual matrix và tolerance.
- [ ] Đo estimator NMSE theo đúng noise scaling được dùng.
- [ ] Kiểm tra actors, message builder, heuristic và online optimizer không đọc true channel.
- [ ] Kiểm tra one-step message queue, sender/mask và request/delivery/drop accounting.
- [ ] Kiểm tra hai action-head log-probabilities, entropy, action masks và GAE truncation.
- [ ] Cost-only task: physical action cố định, message không hữu ích; sender học giảm gửi khi có cost.
- [ ] Positive control: sender biết private binary variable, receiver cần nó ở bước sau; availability cho phép signaling send/no-send.

Nếu correctness fail, sửa rồi chạy lại baseline liên quan. Không dùng số cũ để điền các bảng mới sau thay đổi vật lý.

## P1: thí nghiệm cốt lõi

### A. Main comparison và adaptation

Nhóm controller: USPA no-op; ba static archive points tách riêng; heuristic; random; policy-frozen; MAPPO train no-comm; MAPPO train comm; centralized recurrent PPO; online optimizer.

Khóa initialization, transition budget và codebooks. Centralized actor nhận union của thông tin local, không true CSI; dùng factorized heads. Online optimizer có surrogate/candidate evaluations như resource bổ sung, không true-channel lookup.

**Policy-frozen:** chọn freeze time trên development scenarios, ví dụ 10 và 25 trong episode 100 bước; đây là gợi ý để khóa trước final evaluation, chưa phải config đã chạy. Shared prefix đến freeze time; báo cáo riêng đoạn sau freeze. Khi nói freeze power, phải phân biệt giữ profile rule với giữ realized power vector.

Kết quả chính: adaptive − frozen, CI, fairness/sensing/outage/reconfiguration. Nếu gần bằng nhau, không claim adaptation cần thiết.

### B. Actuator subsets

Train lại bảy nhóm: power; RIS; MA; power+RIS; power+MA; RIS+MA; full. Primary study tắt communication ở tất cả nhóm để cô lập actuator contributions. Các actuator frozen giữ cấu hình khởi tạo như nhau.

Ít nhất 5 independent training seeds cho những so sánh quyết định claim chính nếu thời gian cho phép; hơn 5 tốt hơn khi effect nhỏ. Không biến 50 episodes của một policy thành 50 training replicates.

Kết quả chính: full − MA-only; full − best pair; paired metric differences và uncertainty. Nếu sensing tốt hơn nhưng return ngang nhau, báo cáo trade-off cụ thể.

### C. Communication

Tách train no-comm với train comm/eval off. Corrupt content nhưng giữ sender ID, delivery mask, timing và budget/cost. Chạy cả full graph và harmful graph chọn trên development set. Delivery changes là intervention riêng.

Retrain price free/default/10×; budget 1 và 3. Báo cáo physical quality trước cost, net return, requests, eligibility, delivery, drops, entropy và send probability. Phân biệt request price và delivered-message price.

### D. Observation và recurrence

Error grid: 0, .05, .1, .2, .3. Delay grid: 0, 1, 3, 5. So sánh recurrent/feed-forward được train riêng; masked-CSI được train riêng; fixed/shuffled CSI chỉ là test-time diagnostic. Vary speed và rho để kiểm tra slow-dynamics explanation.

### E. Latency

Cùng hardware/batch/precision; warm-up; đồng bộ GPU. Báo cáo inference, routing/shield, online candidate evaluations, common BD và end-to-end. Median/P95 và số lần đo. Speedup = measured optimizer latency / learned latency với cùng phạm vi timing.

## P2: topology và phân tích mở rộng

Enumerate 64 graphs trên development scenarios; chọn front rồi test held-out. Retrain empty/full/selected graph nếu còn budget. Công bố edge-bit order, objectives và outage constraint. Không gọi frozen-policy optimum là globally optimal topology. NSGA-II chỉ bổ sung nếu có lý do và kiểm tra so với exact front.

## Các đồ thị cần vẽ khi có dữ liệu

| ID | Đồ thị | Trục và uncertainty | Câu hỏi |
|---|---|---|---|
| R1 | Learning curves | transitions / evaluation metric; bands giữa training seeds | Hội tụ có ổn định không? |
| R2 | Adaptive vs frozen | time / min-rate và layout displacement | Continued adaptation có ích không? |
| R3 | Fairness–sensing scatter | min-rate / illumination; CI | Gain có đổi sensing không? |
| R4 | Actuator combinations | subset / metrics với CI | Có complementarity không? |
| R5 | CSI error và delay | error/delay / min-rate; memory variants | Robustness đến từ đâu? |
| R6 | Dynamics sensitivity | speed/rho / adaptive−frozen hoặc recurrent−FF | Delay effect phụ thuộc tốc độ không? |
| R7 | Communication intervention | treatment / paired physical-quality differences | Content hay delivery gây hiệu ứng? |
| R8 | Price response | price / requests, delivery, physical metrics, net return | Sender học phản ứng cost không? |
| R9 | Graph front | rate / sensing / message count | Topology nào đáng giữ với policy này? |
| R10 | Latency | controller / median & P95 ms | Có lợi ích online thực tế không? |

Không vẽ curve giả để minh họa kết quả mong muốn. Với trang giới hạn, đưa R1/R6/chi tiết diagnostics ra supplementary; giữ bằng chứng quyết định claim chính ở main paper.

## Cách điền Results

Mỗi subsection gồm: câu trả lời cho RQ → effect size/CI → đối chứng hỗ trợ → giới hạn diễn giải. Điền đơn vị, training seeds, evaluation scenarios, checkpoint/budget và aggregation của từng bảng. Giá trị thiếu để trống, không thay bằng zero.

Các câu kết luận dự kiến chỉ dùng **nếu dữ liệu hỗ trợ**:

- Q1: “Online adaptation improves [metric] over the policy-frozen control by [effect, CI], supporting an adaptation benefit beyond configuration selection.”
- Q2: “The effect of actor delay depends on [dynamics/memory finding]; the low-level layer still uses current estimated CSI.”
- Q3 negative: “Under the tested semantic payloads, communication does not improve [metric] after controlling for delivery and training, despite successful communication-learning diagnostics.”
- Q3 positive: “Communication improves [metric] under [information/budget condition], but its net utility declines beyond [verified price].”
- Q4: “The full team improves [specific metric] over MA-only and [pair], indicating actuator complementarity under [condition].”

Không điền các câu này bằng kết quả mong đợi. Nếu dữ liệu không ủng hộ coordination, báo cáo giới hạn đó.

## Abstract và format

`abstract_original.txt` giữ nguyên nội dung abstract lấy từ source đã gửi; chưa xác nhận đó là exact text trên OpenReview. Bản abstract trong `main.tex` là working draft có empirical-summary placeholder. Không tự động dùng nó để thay metadata đã đăng ký.

Quy định kiểm tra ngày 02/10/2026: sửa nhỏ title/abstract được phép; thay đổi đáng kể metadata sau abstract deadline không được phép theo quy định thông thường. PDF/supplementary được cập nhật đến 08/10/2026 AoE. Nếu kết quả mới làm abstract cũ sai, hỏi Program Chairs về cập nhật phù hợp.

Nguồn: https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/

PDF hiện tại là bản đọc thử 2 cột dùng `article`, không phải layout nộp AAMAS. Khi hoàn thiện, dùng template chính thức AAMAS 2027, metadata đúng và reference style chính thức; kiểm tra giới hạn 8 trang nội dung. Outline này ưu tiên đầy đủ study design để review trước; có thể chuyển chi tiết diagnostics/config sang supplement để dành chỗ cho đồ thị thật.
