# Baselines

Three white-box DNN testing baselines, run under the same conditions as
BayesWarp: the same input seeds, the same prediction-inconsistency oracle, and
the same per-seed target-model evaluation budget.

```bash
python run_baseline.py --config configs/mnist_lenet5_smoothgrad.yaml --baseline adapt
python run_baseline.py --config configs/cifar10_resnet18_smoothgrad.yaml --baseline nsgen
python run_baseline.py --config configs/mnist_lenet5_smoothgrad.yaml --baseline suntest
```

Each run writes `failures_<baseline>_run<run>.pt` and
`metrics_<baseline>_run<run>.json` to the config's `output_dir`, in the same
format `run_bayeswarp.py` produces, so the existing scripts consume baseline
output unchanged:

```bash
python evaluate_results.py --config <cfg> --failures results/<run>/failures_adapt_run0.pt
python finetune_with_failures.py --config <cfg> --failures results/<run>/failures_adapt_run0.pt
```

## Provenance

| Baseline | Paper | Reference repo |
|---|---|---|
| ADAPT | Lee et al., ISSTA 2020 | https://github.com/kupl/adapt | 
| NSGen | Huang et al., TOSEM | https://github.com/unknownhl/NSGen | 
| SUNTest | Guo et al., TOSEM 2025 | https://github.com/TestingAIGroup/SUNTest |

The reference implementations target TensorFlow/Keras. Two ship no license, so
they are reimplemented from their papers rather than vendored. These are
reimplementations, not the original artifacts. Each module docstring records the
deviations from its source.

