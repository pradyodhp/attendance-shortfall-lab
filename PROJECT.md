# Attendance Shortfall Lab

A reproducible, Colab-friendly machine-learning demo with an interactive attendance recovery calculator.

**This public repository contains independently synthetic data only. No real student names, attendance records, identifiers, photos, real-derived distribution parameters, or private notebook links are included.**

## The problem

A student can be above an overall attendance average while falling below the threshold in one course. This project separates two questions:

1. **Recovery planning:** how many upcoming classes must be attended to reach an example 75% threshold? This is exact arithmetic, not ML.
2. **Forecasting:** can a model classify final attendance shortfall from early attendance observations? Here this is studied only in a clearly labeled simulator.

It is a portfolio and curriculum prototype, not a validated attendance predictor or official exam-eligibility tool. The public demo was rebuilt independently of the private classroom analysis.

## What's included

- `Attendance_Shortfall_Lab.ipynb`: self-contained notebook, synthetic generator, EDA, models, evaluation and an in-notebook dashboard.
- `synthetic_semesters.csv`: 2,400 simulated student-course records for 600 invented students and four invented courses.
- `synthetic_benchmark.csv`: reproducible example evaluation results.
- `requirements.txt`: Python dependencies.

All synthetic rows carry `provenance=independently_synthetic`, generator version, seed and cohort label. There is no identity mapping. The data file is a convenience: the notebook generates the dataset itself.

## Run it

### Google Colab

1. Open the notebook from this repository using Colab's GitHub tab, or download it and choose **File > Upload notebook** in Colab.
2. Use a CPU runtime and **Runtime > Run all**.
3. Scroll to **Interactive recovery calculator**. Change attended, held and remaining classes to explore the recovery scenario.

The notebook is self-contained. No paid API, GPU, credentials or real dataset is required. Widget callbacks need a connected runtime; they are not a deployed website. Colab resources vary and are not guaranteed.

### Local Jupyter

```bash
pip install -r requirements.txt
jupyter notebook Attendance_Shortfall_Lab.ipynb
```

## Approach

The simulator invents four course propensities, a shared student propensity and binomial attendance counts. One observed block and two future blocks form an assumed semester. Future stability, improvement and decline are explicit sensitivity scenarios, not inferred real behavior.

**Target:** final simulated course attendance below 75%.

**Predictors:** first-block course attendance rate/counts, course identity, and other-course attendance summaries available at the same checkpoint. IDs, final attendance and future counts are never predictors.

Students are disjoint across training, validation and test. Preprocessing and SelectKBest are fitted within group-aware cross-validation. Logistic Regression and a shallow Decision Tree are compared with majority-class and current-rate projection baselines. Model and alert threshold are chosen on validation, not test labels.

## Example results: synthetic benchmark only

Seed 42; 600 simulated students; 60/20/20 student-disjoint train/validation/test.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression with cross-course history | 0.846 | 0.845 | 0.797 | 0.820 | 0.920 | 0.906 |
| Decision Tree with cross-course history | 0.802 | 0.775 | 0.778 | 0.776 | 0.900 | 0.876 |
| Majority baseline | 0.558 | 0.000 | 0.000 | 0.000 | 0.500 | 0.442 |
| Current-rate projection | 0.810 | 0.824 | 0.726 | 0.772 | 0.903 | 0.874 |

These results measure how well models recover the generator's assumptions. **They do not establish accuracy on real students or future college attendance.** Versions of dependencies can cause small changes. Full results for both feature sets are in `synthetic_benchmark.csv`.

## Recovery math

For A attended classes out of H held and a threshold p below 1, the minimum consecutive future attendances to recover is:

`max(0, ceil((p*H - A)/(1-p)))`

At the example threshold p = 3/4, this is exactly `max(0, 3*H - 4*A)`.

With R remaining classes, minimum semester-end attendances needed are:

`max(0, ceil(p*(H+R) - A))`

If that exceeds R, attendance alone cannot recover under the simple rule. Medical, event/duty leave, practicals and makeup rules are not silently credited. At 100%, a counted past absence cannot be repaired simply by attending more classes under a no-exemption rule.

## Limits and responsible use

- Simulated examples add no real-world evidence.
- No day-level streaks, causal explanations, grades or excusal decisions are invented.
- Model scores are not guaranteed calibrated probabilities.
- Do not use these scores for punishment, exam exclusion or automated contact.
- A real deployment needs authorized longitudinal data, confirmed counting rules, a completed-semester holdout and local validation.
- Removing names is not sufficient to make classroom attendance safe for public release. Keep real-data notebooks, outputs, widget states and exports out of this repository.

The notebook was executed end-to-end locally before publication. Colab-specific runtime behavior has not been independently tested.

## References

- [scikit-learn: common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: precision-recall and average precision](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Google Colab resource limits](https://research.google.com/colaboratory/faq.html)

No open-source license is selected yet. Public visibility alone does not grant a reuse license.
