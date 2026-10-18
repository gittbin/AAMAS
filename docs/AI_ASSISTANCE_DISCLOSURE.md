# AI assistance record for the working paper

Tool: OpenAI Codex, based on the GPT-6 model family. The exact serving snapshot/version is not exposed in this session and is not invented here. This record describes visible interactions; earlier detailed assistant exchanges are not available in the retained context.

Purpose and affected work: code inspection, correctness fixes and tests; log interpretation; design of development communication diagnostics and screening; design and implementation of fresh final evaluation, independent matched seed blocks, checkpoint selection gates, frozen-policy and actuator comparisons, communication interventions, topology selection and statistical reporting; LaTeX editing, formatting and evidence-backed result export.

The assistant identified the discrete p-value resolution issue for a seven-test Holm family with eight matched blocks and proposed ten blocks before opening new final scenarios. Existing confirmation and development outcomes informed the new protocol; new final scenarios remain closed until all prespecified development selections are locked. The workflow retains outcomes regardless of direction and does not treat smoke tests as scientific evidence. Authors remain responsible for checking design, code, results, citations and manuscript.

Relevant user prompts retained verbatim from this conversation (not a complete historical transcript):

> bạn phân tích log và nghĩ xem có cách nào để cải thiện không

> vậy bắt đầu làm cho tôi

> vậy là cũng có cải tiến rồi, làm bước tiếp theo đi

> vậy là kết quả đã có phần cái thiện giờ hãy phân tích xem bước tiếp theo nên làm gì

> bắt đầu làm đi

> hãy tiếp tục hoàn thiện cho tôi

The current prompt included IDE context for the v4 progress report, working paper PDF and v3 screening report. Exact prompts and assistant outputs from portions of earlier conversations that are not retained are unavailable. Additional interaction records should be preserved rather than reconstructed.

Research figures in this completion workflow use deterministic plotting/code tools with measured data or explicit schematic geometry. No image-generation model was used.

On 2026-10-04, the user again requested a status check: "kiểm tra cho tôi". After all 180 training runs and the locked final opening, the assistant diagnosed an original evaluator failure in the exploratory perfect-CSI grid (feed-forward training seed 305, scenario 12000003, step 0). Float32 inverse-squared estimated gains overflowed during power normalization. The assistant implemented an isolated numerical erratum (`perfect-csi-inverse-power-overflow-erratum-1`) with a source ledger locked before its evaluations: retain finite legacy tensors exactly; use float64 normalization only for nonfinite inverse-power output. The existing gain clamp, codebook, floor and physical model remain the same. Four physical-contract/isolation tests passed, and the three complete pre-failure episodes matched their original traces apart from wall-clock timing. Recovery uniformly reevaluates all 30 perfect-CSI cells in a separate copy, preserves original failure logs, retains primary traces and locked selections, and completes missing evaluation/reference/report work. This correction occurred after final opening and must be disclosed with the exploratory results; it does not constitute new training, seed selection or checkpoint selection.

Policy consulted on 2026-10-04: [submission instructions](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/) and [Q&A](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/). Methodological assistance details belong in the paper or supplementary material; unknown versions/history should be stated. This record is intended for the anonymous reproducibility supplement, subject to author accuracy review.
