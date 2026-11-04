# Experiments

Use `python run.py --list` to list all entry points. Commands run from the project root.

## Main experiments

| Entry point | Description |
|---|---|
| [pushforward](../experiments/pushforward.py) | Evaluate the boundary-mass pushforward law under controlled transformations. |
| [sample_complexity](../experiments/sample_complexity.py) | Measure synthetic learning curves across representation scales. |
| [normal_scaling](../experiments/normal_scaling.py) | Measure label demand under boundary-normal scaling. |
| [tangent_control](../experiments/tangent_control.py) | Measure label demand under boundary-tangent transformations. |
| [backbone_selection](../experiments/backbone_selection.py) | Compare boundary mass and label budgets across frozen backbones. |
| [transferability](../experiments/transferability.py) | Compare transferability baselines and incremental boundary-mass predictions. |
| [near_tie_selection](../experiments/near_tie_selection.py) | Evaluate shortlist selection among closely matched backbones. |
| [multiclass_comparison](../experiments/multiclass_comparison.py) | Compare four multiclass boundary estimators. |
| [acquisition](../experiments/acquisition.py) | Compare three label-acquisition policies. |
| [pilot_refitting](../experiments/pilot_refitting.py) | Measure pilot-label refitting stability. |

## Multiclass comparison

`multiclass_comparison` uses 40 five-way tasks and five stratified splits, shared across four backbones. All backbones use the same semantic class identities. It compares Hard and Smooth OvR/interface estimators using q20 geometric bandwidth, with q10/q30 sensitivity settings. The sigmoid smoothing temperature is 0.05.

```bash
python run.py multiclass_comparison -- --n-tasks 40 --n-seeds 5
```

Outputs: `scores.csv`, `comparison.csv`, and `bandwidth_sensitivity.csv`.

## Acquisition

`acquisition` uses a shared class-balanced 2-spc pilot and budgets of 2, 5, 10, 20, 50, and 100 spc. It evaluates Random, Uncertainty, and Hybrid selection against 95% of the task's best full-probe accuracy. The primary correlation averages each task–backbone pair over five seeds before computing Spearman over 160 pairs. A separate output column records task–seed residual correlation.

```bash
python run.py backbone_selection --output-dir outputs/selection
python run.py acquisition --output-dir outputs/selection -- \
  --n-pairs 40 --n-seeds 5 --pilot-spc 2 --target-ratio 0.95
```

Outputs: `learning_curves.csv`, `budgets.csv`, `summary.csv`, `pilot_scores.csv`, and `acquisition_events.csv`. `--reference-budgets` accepts another full-probe budget table; `--curves` summarizes an existing learning-curve table.

## Pilot refitting

`pilot_refitting` fits five heads for each task, backbone and pilot size (2, 5, or 10 spc), keeping the diagnostic pool fixed. It computes score variability, backbone-order agreement, exact selection agreement and selected-budget agreement. Confidence intervals use 10,000 task-clustered bootstrap samples.

```bash
python run.py backbone_selection --output-dir outputs/selection
python run.py transferability --output-dir outputs/selection
python run.py pilot_refitting --output-dir outputs/selection -- --n-seeds 5 --n-boot 10000
```

`--raw-metrics` and `--selections` accept previously computed experiment outputs for statistical aggregation.

## Score conventions

`utils.boundary.estimate_boundary_mass` implements the paper's two-sided binary coefficient. For signed distance `s` and predicted-class balance `beta`, the hard score is `mean(abs(s) <= epsilon) / (epsilon * beta)`. Its Gaussian version is `2 * mean(phi(s / epsilon)) / (epsilon * beta)`, where `phi` is the standard normal density.

The binary backbone-selection, acquisition, and pilot-refitting tables use one-sided Gaussian surface-density scores and the smaller observed label frequency `alpha_y`. Their kernel score is `mean(phi(s / epsilon)) / (epsilon * alpha_y)`; the companion hard score divides tube probability by `2 * epsilon * alpha_y`. Multiplying either by `2 * alpha_y / beta` gives the corresponding two-sided, predicted-balance score. Regression and policy results use the scores produced by their respective experiment entry points.

Multiclass comparison uses class-versus-leading-rival distances and a top-two interface gate. End-to-end monitoring averages classwise hard and triangular-kernel scores, using observed class frequencies with a 0.01 floor. The standalone MNIST training example uses the canonical two-sided helper with predicted-class balance at each checkpoint.

## End-to-end boundary pressure

`train_end_to_end.py` uses cross entropy plus a boundary-pressure proxy: half the hinge penalty on the true-class-versus-leading-rival logit margin, plus 0.05 times the head's Frobenius norm. The default margin threshold is 0.2. Held-out boundary-mass scores measure the resulting feature geometry.

```bash
python train_end_to_end.py --data-dir data --output-dir outputs/end_to_end \
  --modes standard,boundary_pressure
```

The `boundary_pressure` mode accepts `--lambda-pressure` and `--pressure-lr`. The aliases `psieff`, `--lambda-psi`, and `--psi-lr` support existing command files. Monitoring readouts use test images; prospective backbone-selection diagnostics use independent held-out training features.

For binary MNIST checkpoint measurements:

```bash
python -m experiments.mnist_training --data-dir data --output-dir outputs/mnist --epochs 10
```

## Plotting

After computing the corresponding experiments:

```bash
python run.py normal_scaling --output-dir outputs/selection
python run.py tangent_control --output-dir outputs/selection
python run.py plot_results --output-dir outputs/selection
```

Figures use the normal/tangent curves, backbone budgets and transferability summary in the shared output directory.

## Additional entry points

| Command | Source |
|---|---|
| `python run.py train_custom_probe` | [experiments/train_custom_probe.py](../experiments/train_custom_probe.py) |
| `python run.py train_cifar100_probe` | [train_cifar100_probe.py](../train_cifar100_probe.py) |
| `python run.py train_encoder` | [train_encoder.py](../train_encoder.py) |
| `python run.py train_imagefolder_probe` | [train_imagefolder_probe.py](../train_imagefolder_probe.py) |
| `python run.py evaluate_feature_noise` | [experiments/evaluate_feature_noise.py](../experiments/evaluate_feature_noise.py) |
| `python run.py evaluate_probes` | [experiments/evaluate_probes.py](../experiments/evaluate_probes.py) |
| `python run.py evaluate_image_shifts` | [experiments/evaluate_image_shifts.py](../experiments/evaluate_image_shifts.py) |
| `python run.py sweep_pretrained_probes` | [experiments/sweep_pretrained_probes.py](../experiments/sweep_pretrained_probes.py) |
| `python run.py sweep_encoders` | [experiments/sweep_encoders.py](../experiments/sweep_encoders.py) |
| `python run.py sweep_imagefolder_probes` | [experiments/sweep_imagefolder_probes.py](../experiments/sweep_imagefolder_probes.py) |
| `python run.py sweep_custom_probes` | [experiments/sweep_custom_probes.py](../experiments/sweep_custom_probes.py) |
| `python run.py summarize_boundary_utility` | [experiments/summarize_boundary_utility.py](../experiments/summarize_boundary_utility.py) |
| `python run.py prepare_tiny_imagenet` | [experiments/prepare_tiny_imagenet.py](../experiments/prepare_tiny_imagenet.py) |
| `python run.py summarize_training` | [experiments/summarize_training.py](../experiments/summarize_training.py) |
| `python run.py pushforward` | [experiments/pushforward.py](../experiments/pushforward.py) |
| `python run.py sample_complexity` | [experiments/sample_complexity.py](../experiments/sample_complexity.py) |
| `python run.py boundary_utility` | [experiments/boundary_utility.py](../experiments/boundary_utility.py) |
| `python run.py proxy_comparison` | [experiments/proxy_comparison.py](../experiments/proxy_comparison.py) |
| `python run.py utility_envelope` | [experiments/utility_envelope.py](../experiments/utility_envelope.py) |
| `python run.py multibackbone_proxies` | [experiments/multibackbone_proxies.py](../experiments/multibackbone_proxies.py) |
| `python run.py estimator_cost` | [experiments/estimator_cost.py](../experiments/estimator_cost.py) |
| `python run.py normal_scaling` | [experiments/normal_scaling.py](../experiments/normal_scaling.py) |
| `python run.py feature_noise_control` | [experiments/feature_noise_control.py](../experiments/feature_noise_control.py) |
| `python run.py tangent_control` | [experiments/tangent_control.py](../experiments/tangent_control.py) |
| `python run.py backbone_selection` | [experiments/backbone_selection.py](../experiments/backbone_selection.py) |
| `python run.py estimator_reliability` | [experiments/estimator_reliability.py](../experiments/estimator_reliability.py) |
| `python run.py extract_cifar100_features` | [experiments/extract_cifar100_features.py](../experiments/extract_cifar100_features.py) |
| `python run.py extract_cifar100_siglip` | [experiments/extract_cifar100_siglip.py](../experiments/extract_cifar100_siglip.py) |
| `python run.py transferability` | [experiments/transferability.py](../experiments/transferability.py) |
| `python run.py source_logits` | [experiments/source_logits.py](../experiments/source_logits.py) |
| `python run.py tiny_imagenet` | [experiments/tiny_imagenet.py](../experiments/tiny_imagenet.py) |
| `python run.py extract_tiny_siglip` | [experiments/extract_tiny_siglip.py](../experiments/extract_tiny_siglip.py) |
| `python run.py pilot_selection` | [experiments/pilot_selection.py](../experiments/pilot_selection.py) |
| `python run.py paired_label_savings` | [experiments/paired_label_savings.py](../experiments/paired_label_savings.py) |
| `python run.py pactran` | [experiments/pactran.py](../experiments/pactran.py) |
| `python run.py near_tie_selection` | [experiments/near_tie_selection.py](../experiments/near_tie_selection.py) |
| `python run.py estimator_robustness` | [experiments/estimator_robustness.py](../experiments/estimator_robustness.py) |
| `python run.py multiclass_aggregation` | [experiments/multiclass_aggregation.py](../experiments/multiclass_aggregation.py) |
| `python run.py multiclass_selection` | [experiments/multiclass_selection.py](../experiments/multiclass_selection.py) |
| `python run.py plot_results` | [plot_results.py](../plot_results.py) |
| `python run.py train_end_to_end` | [train_end_to_end.py](../train_end_to_end.py) |
| `python run.py demo` | [demo.py](../demo.py) |
| `python run.py multiclass_comparison` | [experiments/multiclass_comparison.py](../experiments/multiclass_comparison.py) |
| `python run.py acquisition` | [experiments/acquisition.py](../experiments/acquisition.py) |
| `python run.py pilot_refitting` | [experiments/pilot_refitting.py](../experiments/pilot_refitting.py) |
