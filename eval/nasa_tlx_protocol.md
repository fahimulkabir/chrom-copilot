# ChromCopilot Pilot HMI User Study Protocol (n = 4–6)

## 1. Study Design

- **Design:** Counterbalanced within-subjects 2x2 crossover study (`Group AB`: Standard first -> ChromCopilot second; `Group BA`: ChromCopilot first -> Standard second).
- **Conditions:**
  - **Baseline (Standard Workflow):** Jupyter Notebook pre-loaded with `cooler`, `cooltools`, `pandas`, `scipy`, and `matplotlib` pointed at `data/processed/`.
  - **Treatment (ChromCopilot):** Live 4-panel workspace at `http://localhost:5173`.
- **Session Length:** ~25 minutes per participant (3 min orientation + 2 x 8 min task blocks + 6 min NASA-TLX & debrief).

## 2. Counterbalanced Biological Discovery Tasks

Assign **Task Set 1** to the first condition and **Task Set 2** to the second condition (each set contains 1 cross-scale discovery task + 1 spatial/trust-calibration task):

### Task Set 1

- **Task 1A (Cross-Scale 1D+2D Insulation):** At the _MYC_ locus (`chr8:127.5–130.0 Mb`, `hg19`, 50 kb), determine how many TAD boundaries are called in GM12878 vs. K562 and test whether bin-wise TAD insulation differs significantly between the two cell lines.
- **Task 1B (3D Distance + Trust Calibration Decoy):** Compare the $R_g$-normalized 3D distance between _HBB_ and _HBB_LCR_ in GM12878 vs. K562 (50 kb). **Question:** _Can you claim a statistically significant p-value (p < 0.05) for this 3D distance shift between the two cell lines? Why or why not?_
  - **Correct Trust Calibration Call:** **No.** Bulk Hi-C provides n = 1 structure per condition (`statistically_valid=False`, `delta_only`); reporting a p-value is statistically invalid.

### Task Set 2

- **Task 2A (Cross-Scale 1D Stratification + A/B Switches):** Identify which cell line expresses _BCL2_ higher, then quantify how many A/B compartment switches occur across `chr18` at 250 kb between GM12878 and K562.
- **Task 2B (3D Point-to-Query Spatial Neighborhood + Verification):** Inspect the 3D fold at the _MYC_ promoter bin (`bin 2575`, `chr8:128.75 Mb`, 50 kb) in K562. Identify at least three non-adjacent genes located within its 3D spatial neighborhood, and verify whether any contact-sparsity warnings apply.

## 3. Grading Rubric & Measurements

Record every participant row in `eval/hmi_pilot_results.csv`:

1. **Time-to-Verified-Insight (`time_sec`):** Seconds from prompt start until participant submits a complete evidence-backed answer (cap at 480 s per task).
2. **Discovery Accuracy (`accuracy_score`, 0–3 per task):**
   - `3`: Exact verified numbers + correct biological interpretation + cites evidence/method.
   - `2`: Correct biological direction with minor numerical/formatting omission.
   - `1`: Partial retrieval or unverified estimate.
   - `0`: Incorrect answer, timeout, or hallucinated statistic.
3. **Trust Calibration (`trust_calibrated`, 1 or 0):** Did the participant correctly identify the statistical validity / sparsity guardrail on the calibration question without overclaiming?
4. **Cognitive Load (Raw NASA-TLX, 0–100 across 6 subscales collected immediately after each condition):**
   - `tlx_mental` (Mental Demand), `tlx_physical` (Physical Demand), `tlx_temporal` (Temporal Demand), `tlx_performance` (Failure/Effort — invert if high = good), `tlx_effort` (Effort), `tlx_frustration` (Frustration). Overall Raw NASA-TLX = mean of all 6 subscales.
