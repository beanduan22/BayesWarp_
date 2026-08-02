# Artifacts

The repository carries the code, the experiment scripts, the baseline
configurations, and the random seed manifests. The trained model weights are
distributed as release assets because one of them exceeds the GitHub file size
limit.

## In this repository

| Artifact | Location |
|---|---|
| Source code | `src/bayeswarp/`, `baselines/` |
| Experiment scripts | `scripts/`, `run_bayeswarp.py`, `run_baseline.py`, `evaluate_results.py`, `finetune_with_failures.py`, `train.py` |
| Baseline configurations | the `baselines:` block of every config in `configs/` |
| Random seeds | `seeds/` |
| Generated test cases | `samples/` |

## Random seeds

Each file in `seeds/` records the seed set of one dataset, model, and saliency
method: the dataset indices of the selected inputs, their ground-truth labels,
the selection seed, and the five per-run search seeds. The seed set is fixed by
the selection seed and shared across every method and run; only the search seed
changes between runs.

`*_calibration.json` holds the held-out sets used to calibrate the redundancy
thresholds; they are drawn from the same split with `--skip` past the seeds of
the main comparison.

Regenerate with:

```bash
python scripts/export_seeds.py --config configs/<name>.yaml --out seeds/<name>.json
```

ImageNet seed manifests are absent because they require the ImageNet training
split, which is not redistributable.

## Generated test cases

`samples/<config>/` holds `cases.pt` and `manifest.json`. `cases.pt` stores the
unique input seeds once and references them from each generated case, together
with the original prediction, the changed prediction, the target class of the
search, the evaluation index at which the case was produced, and its semantic
consistency score.

```python
import torch
pack = torch.load('samples/cifar10_resnet18_smoothgrad/cases.pt')
seed = pack['seeds'][pack['seed_ref'][0]]
case = pack['cases'][0]
```

Read `manifest.json` before using a pack. The packs currently published are a
reduced-scale preview: they were generated with 20 input seeds and a per-seed
budget of 2000 target-model evaluations, not the 100 seeds and 10000
evaluations of the paper protocol, and they predate the finalisation of the
mutation-magnitude semantics. They are not the artifacts backing the reported
numbers.

Regenerate with:

```bash
python scripts/export_sample_pack.py \
  --config configs/<name>.yaml \
  --failures results/<name>/failures_main_run0.pt \
  --out samples/<name> --count 200 --score-scs
```

## Trained model weights

Published as assets of a release of this repository.

| Asset | Model | Dataset | Size |
|---|---|---|---|
| `mnist_lenet4.pt` | LeNet-4 | MNIST | 0.1 MB |
| `mnist_lenet5.pt` | LeNet-5 | MNIST | 0.2 MB |
| `cifar10_resnet18.pt` | ResNet18 | CIFAR-10 | 42.7 MB |
| `cifar10_vgg16.pt` | VGG16 | CIFAR-10 | 512.3 MB |

`SHA256SUMS` accompanies the assets. Verify with `sha256sum -c SHA256SUMS`.

Place a downloaded asset at the path the config expects:

```bash
mkdir -p results/cifar10_resnet18
mv cifar10_resnet18.pt results/cifar10_resnet18/best.pt
```

The ImageNet subjects (VGG19, ResNet50, EfficientNet-B0) use the torchvision
pretrained weights, so their configs set `checkpoint: null` and no asset is
required.

Every checkpoint is reproducible from the repository without the assets:

```bash
python train.py --config configs/cifar10_resnet18_train.yaml
```
