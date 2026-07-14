# Operational Synergy Between Chronos-Bolt and TightLoop Sentinel

## A Reproducible Edge-Oriented Evaluation of a Neuromorphic Neural Network Engine for Forecast-to-Action Adaptation

Date: 2026-07-09
Status: Zenodo preprint
Author information: Kyuchul LEE, cording.ai

## Intellectual Property Notice

This preprint intentionally reports only the public experimental protocol, black-box inputs and outputs, aggregate metrics, and high-level system role of TightLoop Sentinel. TightLoop Sentinel is described as a neuromorphic neural network engine for operational adaptation. Proprietary implementation details, internal state equations, parameterization, memory layout, and training or update mechanics are intentionally omitted.

## Abstract

Time-series foundation models such as Chronos-Bolt provide useful zero-shot probabilistic forecasts, but operational systems usually need more than predicted values: they need bounded actions such as reserve allocation, alert escalation, buffer adjustment, or capacity staging. This study evaluates whether Chronos-Bolt forecasts can be made more operationally useful by adding TightLoop Sentinel, a neuromorphic neural network engine that converts forecast distributions and recent replay feedback into causal, bounded operational adjustments.

Chronos-Bolt Base was used as the primary forecasting model and Chronos-Bolt Tiny was retained as a comparison model. Four GIFT-Eval-derived tasks were replayed on a Jetson Orin CUDA environment: BizITObs application telemetry, electricity demand, Jena weather, and Bitbrains fast storage traces. The Bitbrains evaluation initially exposed non-finite label horizons; these were handled by removing non-finite forecast/actual rows at trace generation and by adding finite-input guards in the Rust sentinel loader.

Across the Base runs, the combined Chronos-Bolt Base plus TightLoop Sentinel condition improved operational coverage on all four evaluated datasets. Under the default actuator setting, `baseline_tightloop` improved coverage by +32.89 percentage points on BizITObs, +10.07 points on Electricity, +11.50 points on Jena Weather, and +4.14 points on Bitbrains, relative to Chronos-Bolt Base intervals. Mean operational cost decreased substantially on BizITObs (-56.45%), Electricity (-4.26%), and Jena Weather (-7.85%). Bitbrains remained dominated by near-zero interval-width outliers, but no non-finite values remained after the fix and coverage still improved.

These results support a clear separation of roles: Chronos-Bolt estimates future value distributions, while TightLoop Sentinel acts as a neuromorphic neural network engine that translates those distributions into conservative, causal, operation-facing actions. The observed value is not a claim that the sentinel improves the underlying forecast model. The value is that it can improve the forecast-to-action layer where undercoverage, shortfall, reserve waste, and action churn carry direct operational cost.

## Keywords

time-series forecasting, Chronos-Bolt, GIFT-Eval, neuromorphic neural network engine, forecast-to-action adaptation, probabilistic forecasting, operational evaluation, edge AI, Jetson Orin

## 1. Motivation

Chronos-Bolt is a pretrained time-series forecasting model family that can generate probabilistic forecasts in a zero-shot setting. Probabilistic forecasts are useful because they include uncertainty, but an operations team usually needs a second layer that decides what to do with that uncertainty. A data-center, HVAC/BMS, energy, storage, queueing, logistics, or capacity system may ask different questions:

- Should reserve capacity be increased or released?
- Should an alert be escalated, suppressed, or delayed?
- Is the recent forecast miss a one-step disturbance or a regime shift?
- Is the upper tail or lower tail more operationally dangerous?
- How much action is justified without overreacting?

The hypothesis tested here is that Chronos-Bolt and TightLoop Sentinel have complementary roles. Chronos-Bolt provides a forecast distribution. TightLoop Sentinel, as a neuromorphic neural network engine, provides a bounded online action layer that can convert that distribution into operational control signals.

## 2. Related Public Components

Chronos introduced pretrained probabilistic time-series models using tokenized time-series values and transformer architectures. Chronos-Bolt is a faster model family available through the Amazon Chronos ecosystem and Hugging Face model cards. GIFT-Eval is a public benchmark for evaluating general time-series forecasting models across diverse domains.

This draft uses those public components only as black-box forecasting and evaluation inputs. TightLoop Sentinel is not described as a replacement for Chronos-Bolt and is not presented as an anomaly detector. It is evaluated as an operational adaptation layer placed after a forecast model.

## 3. Experimental Environment

The experiments were run in the local `zeroshot` workspace on Jetson Orin.

| Component | Configuration |
| --- | --- |
| Platform | Jetson Orin |
| L4T | R36.5.0 |
| Driver | 540.5.0 |
| CUDA runtime/toolkit | 12.6 |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 |
| Chronos inference | `chronos-forecasting`, CUDA device `cuda:0` |
| cuSPARSELt runtime | 0.6.2.3 Jetson-compatible runtime |
| Sentinel implementation | Rust release binary |

The Chronos CUDA smoke test verified CUDA availability on Orin and successfully ran Chronos inference. The Rust sentinel was built and tested with the local Rust toolchain.

## 4. Data and Forecast Trace Generation

Four public benchmark-derived tasks were selected from the local GIFT-Eval data cache:

| Dataset label in this study | Frequency/task | Rows used |
| --- | --- | ---: |
| `bizitobs_application_10S` | BizITObs application telemetry, short horizon | 1,800 |
| `electricity_D` | Electricity, daily, short horizon, limited to 100 series | 3,000 |
| `jena_weather_10T` | Jena weather, 10-minute, short horizon | 20,160 |
| `bitbrains_fast_storage_H` | Bitbrains fast storage, hourly, short horizon, limited to 1000 series | 46,404 finite rows |

Chronos-Bolt Base was the primary model:

```text
amazon/chronos-bolt-base
```

Chronos-Bolt Tiny was retained as a comparison model:

```text
amazon/chronos-bolt-tiny
```

Forecast traces stored actual values, forecast quantiles from q10 to q90 where available, residuals, interval width, breach indicators, and normalized residuals. The trace generation script now rejects any horizon where the actual value or any available Chronos quantile is non-finite.

## 5. Bitbrains Non-Finite Label Fix

The Bitbrains task initially produced non-finite rows in downstream metrics. The root cause was non-finite horizon labels in the public replay data rather than a TightLoop numerical failure. The fix was applied at two levels:

- Forecast trace generation skips horizons where `actual` or any available Chronos quantile is not finite.
- The Rust sentinel loader skips legacy CSV rows where `actual`, `q10`, `q50`, or `q90` is not finite.

The regenerated Bitbrains traces have the following metadata for both Chronos-Bolt Base and Tiny:

| Model | Candidate rows | Written finite rows | Skipped non-finite rows |
| --- | ---: | ---: | ---: |
| Chronos-Bolt Base | 48,000 | 46,404 | 1,596 |
| Chronos-Bolt Tiny | 48,000 | 46,404 | 1,596 |

String scans of the regenerated Bitbrains forecast trace, TightLoop trace, and operational trace found no remaining `NaN`, `inf`, or `-inf` tokens.

## 6. TightLoop Sentinel Method

TightLoop Sentinel is treated as a neuromorphic neural network engine operating after the forecast model. It consumes the forecast quantile surface, online residual context available from replay, and compact state signals, then emits bounded operational signals.

The public behavior evaluated here is:

- Causal-lag operation: action at a row uses the previous sentinel signal for the same item, not the current residual.
- Continuous actions: the sentinel emits bounded center and reserve adjustment signals rather than hard-coded threshold alerts.
- Distribution-aware operation: lower-tail, upper-tail, center drift, reserve, and regime context are represented separately at the output level.
- Dataset-neutral configuration: the same default actuator weights are applied across all four datasets.

Internal implementation details of the neuromorphic neural network engine are intentionally not disclosed. The important interface is that Chronos-Bolt provides a distribution and TightLoop Sentinel converts the distribution and recent replay state into bounded actions.

## 7. Operational Policies Compared

Three conditions were compared.

| Condition | Meaning |
| --- | --- |
| `baseline` | Chronos-Bolt q10/q50/q90 interval is used unchanged. |
| `tightloop_only` | A causal-lag sentinel center action is applied while the original Chronos interval width is retained. This is a center-action-only sentinel condition, not a separate non-Chronos forecaster. |
| `baseline_tightloop` | Chronos-Bolt interval plus causal-lag sentinel center and reserve actions. This is the primary synergy condition. |

The operational cost is an illustrative replay metric combining normalized interval shortfall, normalized reserve width, and action churn:

```text
operational_cost =
  under_weight * normalized_shortfall
  + reserve_weight * normalized_reserve_width
  + churn_weight * normalized_churn
```

The default weights were held constant across datasets:

| Weight | Value |
| --- | ---: |
| `under_weight` | 8.0 |
| `reserve_weight` | 1.0 |
| `churn_weight` | 0.25 |
| `action_gain` | 1.0 |
| `reserve_gain` | 0.5 |

This cost is not claimed to be the correct business cost for every domain. It is a common replay score used to compare undercoverage reduction, reserve usage, and action stability under the same assumptions.

## 8. Forecast Trace Results

Chronos-Bolt Base generally improved coverage or tail residual behavior relative to Tiny on the selected tasks. The main exception is `electricity_D` p99 normalized residual, where Tiny was slightly lower, while Base still had slightly better 80% coverage and lower mean normalized residual.

| Model | Dataset | Rows | Coverage 80 | Mean normalized absolute residual | p90 normalized residual | p99 normalized residual | Mean absolute residual | Mean 80% interval width |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | BizITObs | 1,800 | 0.3867 | 0.9554 | 2.1427 | 5.1656 | 2034.30 | 1847.05 |
| Base | BizITObs | 1,800 | 0.5600 | 0.5494 | 1.2253 | 1.9630 | 1702.11 | 2324.70 |
| Tiny | Electricity | 3,000 | 0.8173 | 0.3128 | 0.6549 | 1.1722 | 559.87 | 1754.09 |
| Base | Electricity | 3,000 | 0.8280 | 0.3062 | 0.6451 | 1.2538 | 523.08 | 1680.61 |
| Tiny | Jena Weather | 20,160 | 0.8137 | 0.4878 | 0.6851 | 2.0741 | 8.40 | 20.21 |
| Base | Jena Weather | 20,160 | 0.8556 | 0.4052 | 0.5853 | 1.3383 | 6.45 | 24.21 |
| Tiny | Bitbrains | 46,404 | 0.7977 | 434154.1805 | 0.7507 | 7.5645 | 344.01 | 1345.57 |
| Base | Bitbrains | 46,404 | 0.8071 | 372358.6163 | 0.7560 | 5.5494 | 466.58 | 1311.20 |

The Bitbrains mean normalized residual remains extremely large because some interval widths are near zero. For Bitbrains, p90/p99 residuals and coverage are more stable summary statistics than the mean normalized residual.

## 9. Chronos-Bolt Base Operational Results

The primary result table compares the Chronos-Bolt Base baseline to the two sentinel-assisted policies under the default actuator setting.

| Dataset | Policy | Baseline cost | Policy cost | Cost delta | Coverage delta | Shortfall delta | Reserve width delta | p90 cost delta | p99 cost delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BizITObs | `tightloop_only` | 4.1880 | 2.5379 | -39.40% | +18.22pp | -52.36% | +0.00% | -46.24% | -22.99% |
| BizITObs | `baseline_tightloop` | 4.1880 | 1.8239 | -56.45% | +32.89pp | -83.65% | +24.00% | -81.52% | -44.67% |
| Electricity | `tightloop_only` | 1.6193 | 1.4114 | -12.84% | +7.60pp | -44.03% | +0.00% | -55.74% | -12.00% |
| Electricity | `baseline_tightloop` | 1.6193 | 1.5503 | -4.26% | +10.07pp | -58.87% | +13.35% | -29.41% | -28.07% |
| Jena Weather | `tightloop_only` | 3.2589 | 2.9434 | -9.68% | +9.18pp | -15.26% | +1.48% | -48.67% | -10.42% |
| Jena Weather | `baseline_tightloop` | 3.2589 | 3.0031 | -7.85% | +11.50pp | -21.31% | +18.03% | -15.37% | -35.05% |
| Bitbrains | `tightloop_only` | 4166.3823 | 4166.6665 | +0.01% | +1.55pp | +0.01% | +1.69% | +30.14% | +7.67% |
| Bitbrains | `baseline_tightloop` | 4166.3823 | 4166.4771 | +0.00% | +4.14pp | +0.00% | +8.62% | +11.97% | +5.08% |

The strongest synergy appears in BizITObs, where Chronos-Bolt Base plus TightLoop Sentinel reduced mean operational cost by 56.45% and raised interval coverage by 32.89 percentage points. Electricity and Jena Weather also showed improved coverage and reduced shortfall. Bitbrains improved coverage but did not improve mean cost because the cost metric is dominated by near-zero interval-width outliers.

## 10. Tiny Comparison

Chronos-Bolt Tiny remains useful as a lightweight comparison model, but the Base run is the primary reference for future documents.

| Model | Dataset | `baseline_tightloop` cost delta | `baseline_tightloop` coverage delta | `baseline_tightloop` shortfall delta |
| --- | --- | ---: | ---: | ---: |
| Tiny | BizITObs | -45.51% | +20.06pp | -53.69% |
| Base | BizITObs | -56.45% | +32.89pp | -83.65% |
| Tiny | Electricity | -0.39% | +9.03pp | -50.41% |
| Base | Electricity | -4.26% | +10.07pp | -58.87% |
| Tiny | Jena Weather | -8.93% | +14.10pp | -26.70% |
| Base | Jena Weather | -7.85% | +11.50pp | -21.31% |
| Tiny | Bitbrains | -0.00% | +4.83pp | -0.00% |
| Base | Bitbrains | +0.00% | +4.14pp | +0.00% |

Base is clearly stronger on BizITObs and Electricity in this operational replay. Jena Weather and Bitbrains show mixed differences, but Base remains the selected reference because its forecast trace is more stable on the selected public tasks and because future work should avoid optimizing around the smallest model when the target deployment can run Base on Jetson Orin CUDA.

## 11. Exploratory Sentinel Optimization

A small global actuator sweep was performed to test whether TightLoop Sentinel could be better aligned with Chronos-Bolt Base without dataset-specific tuning. The same action and reserve gains were applied to all four datasets.

| Config | Action gain | Reserve gain | Policy | Mean cost delta | Mean coverage delta | Mean shortfall delta | Mean reserve width delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `default` | 1.0 | 0.50 | `baseline_tightloop` | -17.14% | +14.65pp | -40.96% | +16.00% |
| `lower_reserve` | 1.0 | 0.35 | `baseline_tightloop` | -17.27% | +13.62pp | -38.18% | +10.36% |
| `strong_center` | 1.2 | 0.50 | `baseline_tightloop` | -17.48% | +14.74pp | -41.23% | +16.00% |
| `strong_center_lower_reserve` | 1.2 | 0.35 | `baseline_tightloop` | -17.88% | +13.91pp | -38.90% | +10.36% |

The exploratory best average cost was `strong_center_lower_reserve`, which reduced average cost by 17.88% while keeping reserve width growth to 10.36%. This suggests additional headroom for Chronos-specific sentinel calibration. The primary result remains the default actuator because it was the pre-specified common configuration.

## 12. Interpretation

The experimental result should be interpreted as a forecast-to-action improvement, not as a forecast-accuracy improvement. Chronos-Bolt remains the forecasting model. TightLoop Sentinel does not replace Chronos-Bolt and does not claim to know the future better than Chronos-Bolt. Its role is to observe forecast uncertainty and recent replay feedback, then produce bounded operational adjustments.

The synergy is valuable because many real systems care more about operational outcomes than standalone forecast scores:

- An 80% interval with low coverage can cause missed peaks, under-provisioning, or late alerts.
- A too-wide interval can waste reserve capacity or desensitize operators.
- A rapidly flipping adjustment can increase actuator wear or alert fatigue.
- A one-size reserve policy misses the asymmetry between lower-tail and upper-tail risk.

Chronos-Bolt provides a distribution. TightLoop Sentinel uses a neuromorphic neural network engine to convert that distribution into a causal adjustment layer. This creates a practical product path: deploy the best available forecasting model, then add a sentinel that translates forecasts into bounded reserve, alert, buffer, or capacity actions.

## 13. Commercial Relevance

The commercial value is that TightLoop Sentinel can be sold as an add-on layer rather than as a replacement model.

Potential buyers already using forecasting models usually face a final-mile problem: forecasts are available, but operational policies remain hand-tuned. TightLoop Sentinel targets that gap.

Likely value propositions include:

- Lower missed-peak and shortfall cost.
- More reliable forecast intervals under changing regimes.
- Reduced manual threshold tuning.
- More interpretable operation-facing signals such as center drift, tail risk, over-reserve, and regime change.
- Edge deployment with Chronos-Bolt running on CUDA and the sentinel running as a compact Rust operational layer.

The highest-value domains are systems where probabilistic forecasts already influence costly operational decisions: data-center capacity, storage and queue tail control, energy reserve, low-voltage grid planning, HVAC/BMS staging, fleet/logistics buffer allocation, and alert triage.

## 14. Limitations

This is a replay study on selected public benchmark tasks, not a live closed-loop deployment. The operational cost is illustrative and should be replaced by domain-specific business cost before production claims are made.

The GIFT-Eval subset used here is intentionally small enough for rapid Jetson iteration. It should not be presented as a full GIFT-Eval leaderboard result. The study reports trace-level operational metrics rather than official full-benchmark MASE, sMAPE, or weighted quantile loss across all GIFT-Eval tasks.

Bitbrains remains difficult to summarize with mean normalized residuals because near-zero interval widths amplify normalized cost. The non-finite row issue has been fixed, but robust summary metrics and domain-specific cost scaling should be used before making stronger claims.

Finally, TightLoop Sentinel is described only at the public interface level. This protects implementation IP but limits independent reproduction of the exact neuromorphic neural network engine.

## 15. Reproducibility Artifacts

Primary result files in the local workspace:

```text
runs/gift_eval_base_compare/forecast_trace_summary.csv
runs/gift_tightloop_operational_base_summary/combined_operational_summary.csv
runs/gift_tightloop_operational_base_summary/pairwise_operational_compact.csv
runs/gift_tightloop_operational_base_sweep/sweep_compact.csv
runs/gift_eval_forecasts/chronos_bolt_base_bitbrains_fast_storage_H_short_trace_limit1000_meta.csv
runs/gift_eval_forecasts/chronos_bolt_tiny_bitbrains_fast_storage_H_short_trace_limit1000_meta.csv
```

Validation commands:

```text
python3 -m py_compile scripts/gift_eval_dump_forecasts.py
cargo check --manifest-path rust/gift_tightloop/Cargo.toml
cargo build --release --manifest-path rust/gift_tightloop/Cargo.toml
cargo test --manifest-path rust/gift_tightloop/Cargo.toml
```

Non-finite scan:

```text
No NaN/inf/-inf tokens were found in the regenerated Bitbrains Base/Tiny forecast traces,
the regenerated Bitbrains Base TightLoop trace, or the regenerated Bitbrains Base operational trace.
```

## 16. Conclusion

Chronos-Bolt Base provides a strong zero-shot probabilistic forecast baseline on the selected public tasks. TightLoop Sentinel, expressed as a neuromorphic neural network engine, adds a distinct operational capability: it turns forecast distributions into bounded causal actions.

The best evidence for synergy is not a lower forecast residual alone. The evidence is that the combined `baseline_tightloop` condition improved coverage across all evaluated Base datasets and reduced mean operational cost in three of four datasets under a common actuator configuration. The Bitbrains NaN issue was resolved by finite-row filtering and legacy loader guards, leaving a clean replay trace while preserving transparent skip counts.

The next step is a larger full-benchmark replay, followed by domain-specific cost functions and live closed-loop validation in a real operational setting.

## References

1. Ansari, A. F. et al. Chronos: Learning the Language of Time Series. arXiv:2403.07815. https://arxiv.org/abs/2403.07815
2. Amazon Chronos-Bolt Base model card. https://huggingface.co/amazon/chronos-bolt-base
3. Amazon Science Chronos forecasting repository. https://github.com/amazon-science/chronos-forecasting
4. Aksu, T. et al. GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation. arXiv:2410.10393. https://arxiv.org/abs/2410.10393
5. Salesforce AI Research GIFT-Eval repository. https://github.com/SalesforceAIResearch/gift-eval
