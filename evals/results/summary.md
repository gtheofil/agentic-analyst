# Eval results

Generated 2026-07-26 20:12 UTC · 3 briefs · tiers: strong=`gemini-3.6-flash`, fast=`gemini-3.5-flash-lite`

## Aggregate

| Metric | Value |
|---|---|
| Pass rate (hard-check **and** mean ≥ 7) | 0% |
| Hard-check pass rate | 100% |
| Mean coverage | 1.67 / 10 |
| Mean groundedness | 1.00 / 10 |
| Mean actionability | 1.00 / 10 |
| Mean overall | 1.22 / 10 |
| Citation resolution | 100.0% |
| Mean cost per report | $0.0393 |
| Mean judging cost per report | $0.0048 |
| Mean latency | 146s |

## Per brief

| Brief | Domain | Diff. | Hard-check | Cove | Grou | Acti | Mean | Cost | Time |
|---|---|---|---|---|---|---|---|---|---|
| ops-four-day-week | ops | easy | pass | 3 | 1 | 1 | 1.7 | $0.0430 | 143s |
| ops-warehouse-automation | ops | medium | pass | 1 | 1 | 1 | 1.0 | $0.0329 | 128s |
| retail-meal-kits | retail | medium | pass | 1 | 1 | 1 | 1.0 | $0.0420 | 166s |

## Judge reasoning

### ops-four-day-week
- **coverage 3/10** — The report effectively addresses retention and burnout evidence with specific figures from trial studies. However, it fails to include the sample size and execution period for the named trials. It also fails to address the specific billing/utilisation problem inherent to a consultancy business model, merely noting that data on billable hours was absent. Finally, the recommendations lack a concrete trial design and fail to specify what metrics would be measured during the trial.
- **groundedness 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.
- **actionability 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.
- **Not covered:** Findings from a named four-day-week trial, with the sample and the period; The utilisation/billing problem specific to a consultancy, where revenue is billed hours; A trial design concrete enough to run, with what would be measured

### ops-warehouse-automation
- **coverage 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.
- **groundedness 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.
- **actionability 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.

### retail-meal-kits
- **coverage 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.
- **groundedness 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.
- **actionability 1/10** — judge failed: Gemini daily free-tier quota exhausted for gemini-3.6-flash (GenerateRequestsPerDayPerProjectPerModel-FreeTier). This resets at midnight Pacific — pacing will not help. Use a paid key, or point the 'strong' tier at another model in settings.MODEL_TIERS.
