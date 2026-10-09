# SparseWarn2Trade — research plan and paper blueprint (v0.2)
## Learning to detect market manipulation from the few enforcement cases that are actually published

Status: v0.3 results added 2026-10-10 (section R); v0.2 plan text below unchanged; 2026-10-05, rewritten after an eight-lens CCF-A panel review of v0.1 (verdict: reject as written, borderline
after fixes). v0.1 is superseded. Code: `warn2trade/semi/`, `warn2trade/data/{anchors,weak_labelers,semi_dataset}.py`,
`warn2trade/data/regulatory/`, `warn2trade/backtest/{walk_forward,portfolio_eval,event_eval}.py`, `warn2trade/baselines/`,
`configs/{default,smoke}.yaml`, `configs/{baselines,ablations,datasets}/`, `scripts/{run_experiment,power_check,build_anchors,compare_runs}.py`,
`scripts/sweep.sh`, `tests/test_smoke.py` (12 tests). Companion tracks in this checkout: Warn2Trade (`docs/paper_plan.md`,
the trading harness reused here) and CausalGate (`docs/causalgate_research_plan.md`, general leakage taxonomy).
Target: KDD 2027 research track (deadline to verify on the KDD site; early February 2027 is the usual window). AAAI/IJCAI as fallback.

---

## R. Results so far (v0.3, 2026-10-10; all runs via cq, project home sparsew2t, code github.com/SuuTTT/sparsew2t)

Event-level metrics pooled over folds and seeds, case-bootstrap 95 % CIs, paired bootstrap for differences
(scripts/aggregate_sweep.py). Thresholds from the validation fold; evaluation on the oracle anchor raster.

**Synthetic (104 runs, 41 test cases, 4 seeds).** Event AUROC saturates (about 0.999) for every anchor-trained method;
Deep SAD has the best event AP (0.888 at K = 1, 0.965 at K = 3) vs ours (0.727, 0.877). Weak-only 0.75, label model with causal
LFs 0.48. The generator plants a strong pre-onset volume ramp, so this panel is a plumbing check, not evidence.

**Crypto pump-and-dump, La Morgia et al. (180 runs, 277 test cases, 84 coins, hourly, 4 seeds).**

| K | ours | Deep SAD | Supervised-K | FixMatch | Weak-only | ours - Deep SAD (95 % CI) |
|---|---|---|---|---|---|---|
| 1 | 0.849 | 0.951 | 0.896 | 0.897 | 0.716 | -0.102 (-0.119, -0.083) |
| 3 | 0.888 | 0.932 | 0.910 | 0.909 | 0.748 | -0.044 (-0.059, -0.033) |
| 10 | 0.875 | 0.948 | 0.953 | 0.956 | 0.695 | -0.073 (-0.087, -0.058) |
| 30 | 0.886 | 0.957 | 0.953 | 0.954 | 0.738 | -0.071 (-0.083, -0.057) |
| all (~79) | 0.963 | 0.967 | 0.966 | 0.968 | 0.730 | -0.004 (-0.010, 0.003) |

Findings: (1) a handful of real cases beats anchor-free weak supervision by 0.13-0.23 AUROC (supported); (2) the full method is
significantly WORSE than plain anchor-trained baselines at K <= 30 and raises 3-40x more false alarms (79-114 vs 2-38 per 1,000
normal bars); it only ties with all cases. Suspected cause: the weak-label teacher (LFs fire on any large move, not on pumps).

**CSRC pilot (auto-extracted, unverified cases; 344 tight cases, 5,408 A-shares, daily).** Power check GO: 350 pooled test cases,
MDE 0.042 paired AUROC. Release vs naive availability halves the usable training cases (e.g. fold 9: 50 vs 102). One smoke fold
(K = 3): event AUROC 0.935 (0.903-0.965), recall 0.63, 27 false alarms per 1,000, median lead 37 trading days. Full batch: in
progress (32 of 144 runs done at the time of writing).

**Crypto ablations:** in progress.

**Consequence for the paper.** The method claim ("our anchored weak-to-semi-supervised detector beats baselines") is contradicted
on crypto. Two routes: (A) reposition as a benchmark-and-protocol paper (enforcement-lag protocol, verified CSRC case set,
"a few real cases beat weak supervision; semi-supervision on top does not help") for a KDD ADS or datasets-and-benchmarks track;
(B) redesign the method after the ablations identify the harmful component, then rerun. Route A is the shorter path.

**Engineering notes.** A conversion of queued runs to GPU jobs failed because the env has CPU-only PyTorch; those runs are being
resubmitted as CPU jobs. cq did not copy some verified outputs back to the home master (j-1007-febf4c, j-1007-27a522) and never
propagated a single 1.1 GB file; mitigated by 50x smaller score files and gzip parts.

## 0. Review response matrix (v0.1 → v0.2)

| # | Gap found by the panel | Fix | Status |
|---|---|---|---|
| 1 | Evaluation scored against the builders' rule labels, i.e. the proxy the weak supervision imitates (circular) | Evaluation labels = oracle anchor raster (`eval.labels: anchors`); rule labels reported separately as "proxy events"; synthetic fixture uses rule proxies so the two differ; test asserts every scored row carries the anchor label | **Done (code + test)** |
| 2 | Thresholds and model choices made on the test fold; validation fold unused | Threshold = best F1 on validation if it holds ≥ `min_val_events` cases, else the validation fire quantile; source logged per run | **Done** |
| 3 | Label budget collapsed (K = 5 ≡ Full); silent capping; draw coupled to the training seed | Nested per-class case sampling keyed on case id and a dedicated `budget_seed`; K counted in cases; `K_eff`, `n_available`, `capped` recorded per run; anchors must intersect the training window; negatives shaped only by *sampled* cases | **Done (code + test)** |
| 4 | Deep SAD scored by an untrained head; DevNet mixed with SAD; FixMatch taught by the labelling functions; baselines on different folds; sweep aborted | Deep SAD scored by ‖z − c‖²; DevNet = deviation loss only; FixMatch / FlexMatch self-teach with no LFs; every detector traded by one shared validation-threshold rule; sweep passes one dataset overlay, one budget seed, identical folds and seeds to every method and logs failures instead of aborting | **Done (code + e2e test for Deep SAD)** |
| 5 | Conformal gate bypassed for mixture positives; calibrated on training normals; PLCA bound vacuous | Gate now applies to every teacher positive, calibration rows are held out and never trained on, BH replaces the per-row threshold, score fitted symmetrically. **Then measured:** valid gate selects nothing at ~1 % prevalence; power-restoring trimming gives FDR 0.71 at q = 0.1. The gate is **removed from the contributions**, kept as an ablation, and the negative result is reported (§3.4) | **Done; claim withdrawn** |
| 6 | Theory: Prop. 1 is folklore without conditions; unproven O((K+n)^{-1/2}) rate; dependent bars counted as samples | Lemma 1 (identifiability up to the mirror) cited with its conditions; rate deleted; **Theorem 1** (anchor basin selection, exponential in the number of independent *cases*, any within-case dependence) proved and checked by simulation | **Done (theory + test)** |
| 7 | "Executable alpha" unsupported: exit-booked, un-netted PnL; no benchmark or null; no borrow | Netted mark-to-market portfolio with latency, tradability, per-slot costs and borrow fees; long-only de-risk overlay against an equal-weight benchmark (headline); timing-permutation null; DSR across all configurations. Claim narrowed to "cost-adjusted monetisability" | **Done (code + test)** |
| 8 | No crisp contribution; closed loop belongs to Warn2Trade; release protocol overlaps CausalGate | Contributions rewritten (§1); closed loop cited as Warn2Trade's harness; release-date availability used as *the evaluation protocol* and credited to the CausalGate taxonomy (its channel L5); our protocol contribution is the enforcement-lag measurement on real regulators | **Done (plan); portfolio decision below** |
| 9 | Universe excluded the manipulated issuers; seven of ten datasets had no enforcement labels | Scope cut to three datasets with real anchors (§5); vendor-agnostic daily-bars builder with A-share price limits; sheet→anchors converter with per-class match report; late releases no longer clipped into the panel | **Code done; data curation needs you** |
| 10 | Statistical power unknown with K ≤ 10 and few folds | Event-level metrics pooled across folds with case-bootstrap CIs and paired comparisons; `power_check.py` go/no-go with a minimum-detectable-effect calculation | **Done (code + test)** |

Decisions that are yours, not the code's: (a) the CSRC and SEC case curation (two verifiers per case; template
`docs/sparse_anchor_sheet_template.csv`); (b) the data licences (Tushare/CSMAR for A-shares; CRSP or an OTC-inclusive vendor for
the US); (c) whether CausalGate or this paper reports the naive-vs-release leakage numbers on enforcement labels (§1, last paragraph).

---

## 1. Titles, abstract, contributions

**T1.** *Published Too Late to Learn From? Anchored Weak Supervision for Market-Manipulation Detection with a Handful of Enforcement Cases*
**T2.** *Anchoring Weak Supervision with Enforcement Cases: Label-Efficient Manipulation Detection under Release-Date Constraints*
**T3.** *How Many Enforcement Cases Does a Manipulation Detector Need?*

**Abstract (~220 words).** The only trustworthy labels for market manipulation are regulator enforcement cases. They are few,
and they are published one to four years after the conduct, so a detector deployed at time t may learn only from cases already
public at t. Cheap market proxies (trading halts, price-limit hits, abnormal moves, volume and spread bursts) are abundant but
noisy, and unsupervised label models that combine them must *assume* the sign of every proxy's accuracy. We show that a few
published cases are enough to fix that sign: in a conditionally independent label model the probability of selecting the
mirrored solution decays exponentially in the number of independent cases, whatever the dependence between bars inside a case.
We build on this an anchored weak-to-semi-supervised detector and evaluate it under a release-date-aware walk-forward protocol on
China A-shares with CSRC sanction decisions, US equities including OTC issuers with SEC litigation releases, and [crypto
pump-and-dump as a label-rich control]. Detection is scored against the enforcement cases themselves, at the event level, with
case-bootstrap confidence intervals, as a function of the number of cases actually available. We also report how warnings
translate into a long-only de-risking overlay after costs, against an equal-weight benchmark and a timing-permutation null.
[Numbers to be filled; no result is claimed before the runs.]

**Contributions.**
1. *Problem and evaluation.* Manipulation detection with enforcement labels restricted to cases published by decision time,
   scored at the case level on the cases themselves; a measured enforcement-lag profile for two regulators (lag quantiles per
   anomaly class) that determines how many cases a realistic model can see.
2. *Method and theory.* Anchored weak supervision: anchors replace the α_j > ½ assumption of label models. Theorem 1 gives the
   number of independent cases needed to select the correct basin with probability 1 − δ; we document that EM converges to the
   mirrored basin without anchors (reproduced in `tests/test_smoke.py`).
3. *Label-efficiency evidence.* Event-level detection versus the number of available cases (K = 1, 2, 3, 5, 10, all), nested,
   against faithful Deep SAD, DevNet, FixMatch/FlexMatch, Snorkel-style weak supervision without anchors, and the label model
   alone, with paired case-bootstrap tests; plus cost-adjusted monetisability of the warnings.

Not claimed: the closed-loop utility training (Warn2Trade's contribution; used here as an ablation), the conformal gate (a
documented negative result, §3.4), "state of the art", "first", or "alpha".

Positioning against CausalGate: CausalGate lists availability lag (its channel L5, including AAER release dates) as one leakage
channel in a general taxonomy. Here the release-date protocol is the *evaluation protocol*, credited to that taxonomy and to Bao
et al. (JAR 2020), who already impose a gap because AAERs are published late. If both papers are submitted, only one should
report the naive-vs-release inflation on enforcement labels; this plan assumes CausalGate reports it and this paper reports the
lag profile and the label-efficiency curve.

---

## 2. Problem formulation

Panel: instruments i ∈ [N], bars t ∈ [T]; decision point (t, i) with a multimodal window x_{t,i}, realised volatility σ_{t,i},
impact curve I_{t,i}(h) = log p_{t+h} − log p_t (target and backtest input only).

Case k: (i_k, τ_k^on, τ_k^peak, τ_k^end, τ_k^rel, c_k) = instrument, conduct window, peak bar, publication bar, class. Warning
window [τ^on − lead, τ^peak].

Availability: 𝒜_rel(t) = {k : τ_k^rel ≤ t} (realistic, default); 𝒜_naive(t) = {k : τ_k^end ≤ t} (leaky, audit only);
𝒜_oracle = all cases (evaluation labels; "ceiling" runs are labelled as such).

Label budget for a fold with training window [t_lo, t₀]: take the cases in 𝒜_rel(t₀) whose conduct window intersects
[t_lo, t₀]; for each class take the first K in a fixed pseudo-random order keyed on (budget seed, case id), so budgets are nested
across K and across folds. Positives = their warning windows; confident normals = bars more than `neg_margin` bars from any
*sampled* case and from any labelling-function positive. Everything else is unlabelled. A calibration set (block-level uniform
draw, ≤ 1 row per asset × `cal_block` bars) is held out from every loss (used only by the gate ablation).

Evaluation labels = the oracle anchor raster. The trainer never reads them; they are used for metrics and for reporting PLCA.

Labelling functions λ_j(x) ∈ {−1, 0, +1} may look ahead ℓ_j bars (they are targets, not features); the purge is raised
automatically to max(ℓ_j, max h + latency + exec_max). Test-time scoring of weak-supervision baselines uses only causal LFs.

---

## 3. Method

### 3.1 Anchored label model

Class prior π; LF j abstains with probability 1 − β_j and, when it votes, agrees with y with probability α_j; LFs independent
given y. Posterior logit P(y = 1 | Λ) = log(π/(1−π)) + Σ_{j: Λ_j ≠ 0} Λ_j log(α_j/(1−α_j)). EM over all training rows with
anchors and confident normals as observed y and a Beta(a₀, b₀) prior on α_j (`semi/label_model.py`).

**Lemma 1 (identifiability up to the mirror; Dawid & Skene 1979; Allman, Matias & Rhodes, Ann. Statist. 2009; Ratner et al.
NeurIPS 2016; Fu et al. ICML 2020).** With J ≥ 3 conditionally independent LFs, α_j ≠ ½ and non-degenerate coverage, the vote
distribution identifies (π, α, β) up to the joint relabelling (α_j ↦ 1 − α_j ∀j, π ↦ 1 − π). Without labels the
mirror is resolved only by assuming α_j > ½ for all j. *(Cited, not claimed.)*

**Anchor basin selection.** After EM, compute the anchor log-likelihood of the fitted accuracies and of their mirror and keep
the larger; initialise α from the anchors (`AnchoredLabelModel.fit`, flag `flipped_`).

**Theorem 1 (cases needed to select the right basin).** Let EM return either the true parameters or their mirror (Lemma 1),
and let the anchors consist of K cases that are mutually independent, each contributing n labelled rows whose votes may be
arbitrarily dependent inside the case. Write w_j = log(α_j/(1−α_j)) > 0 in the true orientation,
μ = n Σ_j β_j (2α_j − 1) w_j and B = n Σ_j w_j. Then the selection rule picks the wrong basin with probability at most

  exp(−K μ² / (2B²)) = exp(−K ρ² / 2),  ρ = Σ_j β_j (2α_j − 1) w_j / Σ_j w_j ∈ (0, 1],

so K ≥ 2 log(1/δ) / ρ² cases suffice for confidence 1 − δ, independently of n and of the size of the unlabelled pool.

*Proof.* For fitted accuracies a, the rule compares ℓ(a) = Σ_m Σ_{j covers m} [s_mj log a_j + (1 − s_mj) log(1 − a_j)] with
ℓ(1 − a), s_mj = 1 if LF j agrees with the anchor label. ℓ(a) − ℓ(1 − a) = Σ_m Σ_j (2s_mj − 1) w_j(a) = ±D, with
D = Σ_k D_k computed with the true-orientation weights; in both basins the rule is correct iff D > 0. Each D_k ∈ [−B, B] and
E D_k ≥ μ (conditional independence of LFs given y and the agreement probability α_j hold per row; dependence between rows does
not change the expectation). The D_k are independent across cases, so Hoeffding's inequality gives
P(D ≤ 0) ≤ exp(−2(Kμ)² / (K(2B)²)) = exp(−Kμ²/(2B²)). ∎

Remarks. (i) Counting bars instead of cases would overstate the evidence by a factor n; the bound uses cases. (ii) ρ is
estimable from the anchors, so `semi.label_model.cases_needed` reports the K required for a target δ. (iii) The simulation in
`test_theorem1_basin_selection_bound_holds_and_decays` uses strongly dependent rows within a case and confirms the bound.
(iv) The theorem says nothing about *how good* the posterior is once the basin is right; that is what the label-efficiency
curve measures, and the `sign_only` ablation isolates what anchors buy beyond the sign.

### 3.2 Detector training

Encoder f_θ (multimodal fusion), score head g_φ, hypersphere centre c, extrapolator and policy from the Warn2Trade harness.
Stages (`semi/trainer.py`): A, self-supervision (NT-Xent between weak/strong views + masked reconstruction) on all rows;
W (warm), L_sup on anchors and confident normals + L_ssl; B, consistency on the unlabelled pool with teacher
p_t = γ(e) P̃(y=1 | Λ) + (1 − γ(e)) σ(g_φ(f_θ(weak view))), FlexMatch class-wise thresholds, outcome-consistency weights on
teacher positives; C, optional joint stage adding Warn2Trade's differentiable trade utility (ablation `no_trade_loss` reports its
contribution; it is not a claimed contribution).

  L_sup = λ_sad · [(1−y)‖z−c‖² + y η (‖z−c‖² + ε)⁻¹] + λ_dev · [(1−y)|δ| + y max(0, m − δ)],  δ = standardised score
  L_cons = Σ_x ω(x) 1[conf ≥ τ_t(ŷ)] BCE(g_φ(f_θ(strong view)), ŷ) / Σ ω 1[·]
  ω(x) = floor + (1 − floor) σ(κ (max_h |I_x(h)| / (σ_x √h) − m_out)) for teacher positives (training windows only)

Anchors enter twice: through L_sup (they do train the encoder; v0.1 §10.1 was wrong on this) and through the label model.

### 3.3 What we measured about a conformal gate (reported as a negative result)

Proposition 2 (Bates et al., Ann. Statist. 2023; Marandon et al., Ann. Statist. 2024). If C₀ is drawn uniformly from the
unlabelled pool, the gate score is a symmetric function of C₀ ∪ D_u, normal rows are exchangeable and anomalies are
stochastically larger, then conformal p-values are valid and PRDS and BH(q) controls the FDR of the selected pseudo-positives
at q π₀. Feasibility: with m unlabelled and n calibration rows, BH can select k rows only if k ≥ m / (q (n + 1)), and only if
those k rows beat every calibration score (logged as `gate_k_min`).

Measured on the default synthetic configuration (one fold, warm stage, `scratchpad` diagnostic reproduced by
`configs/ablations/conformal_gate*.yaml`): label-model gate score AUROC 0.98; valid gate (no trimming) selects 0 rows because the
~0.7 % anomalies in C₀ occupy its top; trimming C₀ by the label-model posterior selects 195 rows with recall 0.95 but empirical
FDR 0.71 at q = 0.1, because trimming removes exactly the high-scoring normals. Conclusion: at manipulation-level prevalence a
*valid* FDR gate on weak labels is powerless and a *powerful* one is invalid. The paper reports this trade-off in the ablation
section; the default method does not use the gate. Contaminated-reference conformal methods (Bashari, Epstein, Romano & Sesia,
2025†) are the right literature to cite and a possible future direction.

### 3.4 Score-to-trade (shared by every detector)

Threshold from validation; on fire, trade against the trailing move (fade) with weight w for `hold_h` bars, or stay out of the
asset (overlay). Our learned policy is reported additionally (`rule = policy`), never as the only comparison.

---

## 4. Related work to cite (v0.1 omitted these)

Weak supervision and label models: Dawid & Skene 1979; Ratner et al. 2016 (data programming), 2017 (Snorkel, VLDB); Fu et al.
2020 (FlyingSquid, ICML); Allman, Matias & Rhodes 2009; Zhang et al. 2021 (WRENCH, NeurIPS D&B). Weak supervision + self-training
with rules: Karamanolakis et al. 2021 (ASTRA, NAACL). Semi-supervised and few-label AD: Ruff et al. 2020 (Deep SAD, ICLR); Pang et
al. 2019 (DevNet, KDD) and 2023 (PReNet, KDD); SPADE (TMLR 2023†); Sohn et al. 2020 (FixMatch); Zhang et al. 2021 (FlexMatch).
Conformal outlier detection: Bates et al. 2023; Marandon et al. 2024 (AdaDetect); Bashari et al. 2025† (contaminated reference).
Finance: DeLise 2023 (Deep SAD on TMX futures); Bao et al. 2020 (AAER fraud, JAR); Lin & Yang 2025 and Poutré et al. 2024
(synthetic / unsupervised LOB manipulation); La Morgia et al. 2023 (crypto P&D); Cao et al. 2026 (7-token rug pulls); leakage:
López de Prado 2018, Bailey & López de Prado 2014 (DSR), Kapoor & Narayanan 2023, CausalGate (sibling). † = verify authors and
venue before citing. All finance items: see `paper_catalogue.md` for ids.

---

## 5. Datasets (cut from ten to three; all need real anchors)

| ID | Market / universe | Anchors | Bars | Status |
|---|---|---|---|---|
| **D-CN** (primary) | All A-shares incl. delisted, ST, ChiNext, STAR, BSE (Tushare / AkShare / CSMAR export → `daily_bars`, `market: cn`, price-limit freezing) | CSRC 行政处罚决定书: market manipulation (操纵), false declaration (虚假申报), insider trading as detection-only; day-resolution conduct windows, publication date | daily, 2010–2025 | builder + converter done; **curation and data export needed** |
| **D-US** | US common stock incl. OTC and delisted (CRSP if licensed; else an OTC-inclusive vendor) | SEC litigation releases: pump-and-dump, matched / wash trades; insider trading detection-only; month-resolution windows (sensitivity to ±20-bar jitter reported) | daily, 2008–2025 | same builder; **vendor choice and curation needed**; dropped if no OTC data |
| **D-CRYPTO** (label-rich control) | Binance spot, coins in the La Morgia et al. Telegram pump list | announced pumps (hundreds) → full K curve and an honest "Full" | 1-minute | needs a minute-bar builder (Warn2Trade's `crypto_pd` is a stub); synthetic planted anchors serve as the control until then |

Dropped from the paper (appendix at most): StockNet (≈ 0 cases on 88 large caps), LOBSTER spoofing (no public labels), FNSPID/EDGAR
(accounting fraud is a different object), Qlib-Alpha158, flash crashes, Elliptic++ (labels are not scarce).

Expected scale, to be confirmed by `scripts/power_check.py` before any GPU time: CSRC publishes a few dozen manipulation and
insider decisions per year; SEC litigation releases on pump-and-dump and matched trading number in the tens per year. Pooled over
ten annual test folds that is O(100) test cases on D-CN, enough for case-bootstrap CIs; a NO-GO (< 30 pooled cases) moves the
dataset to detection-only.

---

## 6. Baselines (all on identical folds, budgets, seeds, thresholds and trade rule)

| Method | Config | Score |
|---|---|---|
| Deep SAD (Ruff et al. 2020; DeLise 2023 setting) | `configs/baselines/deep_sad.yaml` | ‖z − c‖² |
| DevNet (Pang et al. 2019) | `devnet.yaml` | head |
| FixMatch (Sohn et al. 2020), FlexMatch (Zhang et al. 2021) | `fixmatch_fin.yaml`, `flexmatch_fin.yaml` | head |
| Supervised-K (anchors + normals only) | `supervised_k.yaml` | head |
| Weak-only, Snorkel-style (no anchors) | `weak_only.yaml` | head |
| Label model alone (causal LFs at test time) | `label_model_only.yaml` | posterior |
| Isolation Forest | `isolation_forest.yaml` | −score_samples |
| To wrap from upstream: PReNet, ASTRA, a TS foundation model with a few-shot head (UniTS or One-Fits-All) | — | — |

---

## 7. Metrics

Primary (event level, pooled over folds, case-bootstrap 95 % CIs, `backtest/event_eval.py`): event AUROC, event AP, event recall
at the validation threshold, false alarms per 1,000 normal bars, median lead (bars before peak). Paired differences against each
baseline with one-sided bootstrap p-values (`scripts/compare_runs.py`).
Secondary (fold level): AP, AUROC, R@K, F1 at the validation threshold, WAD, PLCA (pseudo-label precision against cases).
Label efficiency: the curve of event AUROC against K_eff (cases actually used), LES slope, area under the curve, K₈₀.
Monetisability (`backtest/portfolio_eval.py`): long-only de-risk overlay active return and information ratio against equal weight
(headline), with a timing-permutation p-value; netted fade strategy with borrow fees (secondary); DSR across all configurations;
cost curves 1–30 bps with break-even cost.

---

## 8. Experiments

0. **Power gate.** `power_check.py` per dataset; go only if ≥ 30 pooled test cases.
1. **E1 headline figure.** Event AUROC (and overlay IR) vs K_eff ∈ {1, 2, 3, 5, 10, all}, release protocol, ours vs weak-only,
   Deep SAD, FixMatch, label-model-only, with CIs; D-CN and D-US; D-CRYPTO adds the full curve.
2. **E2 lag profile.** Enforcement-lag quantiles per regulator and class; cases visible per fold under release vs naive; oracle
   ceiling. (Naive-vs-release *performance* inflation only if CausalGate does not report it.)
3. **E3 main table** at K = 3 and K = all: primary metrics + overlay IR + permutation p.
4. **E4 ablations** (`configs/ablations/`): sign_only (what anchors buy beyond the sign), no_anchors, no_label_model,
   teacher_labelmodel, no_consistency, no_ssl, no_warm, no_outcome_weights, no_trade_loss / two_stage / plugin_policy, conformal_gate
   and conformal_gate_trimmed (the §3.3 trade-off), rule_label_eval (how much evaluating on proxies flatters every method).
5. **E5 robustness.** Anchor date jitter ±{0, 5, 20} bars (`jitter_anchor_dates`), one-LF-out, neg_margin, latency, budget seed.
6. **E6 monetisability.** Cost curves, borrow sensitivity, overlay vs fade, DSR.

---

## 9. Implementation status (verified 2026-10-05, CPU)

`python3 tests/test_smoke.py` → 12/12 pass: label-model anchors and mirror, Theorem 1 bound under within-case dependence,
conformal validity and BH FDR, gate on teacher positives, nested budgets, evaluation on anchors with held-out calibration, netted
backtest accounting and borrow, event bootstrap and paired deltas, daily-bars builder with A-share limit freezing and the
sheet converter (late release not clipped), end-to-end runs of the method, Deep SAD and a score-only baseline.

Smoke / synthetic runs are plumbing checks, not results: synthetic events are easy at the event level (event AUROC ≈ 1), the
fold-level AUROC of the head is 0.6–0.7, and the overlay loses against equal weight at this toy scale.

```bash
python3 tests/test_smoke.py
```
```bash
python3 scripts/run_experiment.py --config configs/smoke.yaml
```
```bash
python3 scripts/power_check.py --dataset-config configs/datasets/csrc_sanctions.yaml
```
```bash
bash scripts/sweep.sh configs/datasets/csrc_sanctions.yaml
```

Not done (and why): case curation (human verification; template provided), vendor exports and licences, the crypto minute-bar
builder, wrappers for PReNet / ASTRA / a foundation model, and every real-data run. No number in this plan is a result.

---

## 10. Timeline (17 weeks to an early-February 2027 deadline; critical path in bold)

Weeks 1–4: **CSRC sheet (≥ 100 decisions, two verifiers)**, A-share export, `build_anchors.py`, power check. Weeks 3–6: SEC
sheet and OTC data decision; crypto minute builder. Weeks 5–9: E1–E3 on D-CN, then D-US / D-CRYPTO. Weeks 8–11: ablations,
robustness, monetisability. Weeks 11–14: writing; weeks 15–17: buffer, reproducibility package. Compute: the reduced suite is
≈ 3 datasets × (1 + 8 baselines + 16 ablations) × 6 budgets × 4 seeds × ≤ 10 folds; daily panels run in minutes per fold on one
GPU; budget about one GPU-week with the ablations restricted to K ∈ {3, 10}.

---

## 11. Risks, ethics, reproducibility

- **Power.** If D-US yields < 30 pooled cases it becomes detection-only; the paper then rests on D-CN plus the crypto control.
- **Selection bias of enforcement.** Cases are caught manipulations; anchors are high-precision, low-recall positives and are
  never used as negatives; confident normals keep a margin from every sampled case.
- **Proxy dependence.** Rule labels are reported only as "proxy events" and as an ablation of the evaluation target.
- **Dual use.** The trading section is framed as de-risking (selling out of suspected manipulation), not as front-running
  manipulators; no detector output is released for live tickers.
- **Reproducibility.** Anchor CSVs with source URLs and both verifiers' initials; match reports; LF definitions; budget seed and
  K_eff per run; all configs, four seeds, per-fold rows, scores per test row, DSR trial counts; 12 tests.
