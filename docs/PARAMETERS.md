# Model settings

## Architecture

Input shape is `[B,62,5]`. One graph-convolution block contains two parallel Chebyshev terms, `T0=I` and `T1=-D^(-1/2) A D^(-1/2)` (maximum order `K=1`). Each branch applies graph filtering, a KAN activation, and a `Linear(5,32)` projection. Branch outputs are added to produce `[B,62,32]`.

The adjacency is `A=ReLU((W+W.T)/2)`. The shared `62x62` parameter matrix is initialized from electrode distances with delta=5, threshold=0.1, and diagonal entries equal to one. The matrix is shared across samples and updated during training, except in the frozen-adjacency ablation.

The classifier is:

```text
Flatten(62x32=1984)
Dropout(p=0.5)
Linear(1984,128)
ReLU
BatchNorm1d(128, eps=1e-5, momentum=0.1)
Linear(128,C)
```

The output contains class logits. `C=3` for SEED/EEG-Emotion and `C=4` for SEED-IV.

## Activations and ablations

| Configuration | Definition | Trainable parameters, C=3 |
|---|---|---:|
| 5-Swish | Five KAT groups, Swish initialization | 259003 |
| 5-B-spline | Five band-shared cubic splines, retaining the linear projections | 259051 |
| 5-Identity | Five KAT groups, identity initialization | 259003 |
| 1-Swish | One KAT group shared across frequency bands | 258971 |
| w/o KAN | Each branch uses Linear(5,32), ReLU, Linear(32,32) | 261063 |
| w/o Dyn-Adj | Initial adjacency is frozen | 255159 |
| w/o group | Independent rational coefficients for every input-output edge | 262151 |

SEED-IV adds 129 classifier parameters. The frozen-adjacency model stores 259003 parameters, including 3844 non-trainable adjacency entries.

The installed KAT module uses numerator degree 5 and denominator degree 4. Each branch shares six numerator coefficients across groups and has four denominator coefficients per group. Branches have independent parameters. Projection weights use normal initialization with standard deviation `sqrt(2.817820527/5)` for Swish and `sqrt(1/5)` for Identity; biases are zero. Classifier weights use Xavier normal initialization.

B-spline uses a fixed core grid over `[-1,1]` with five intervals and three extended knots at each end. Each band has eight cubic-spline coefficients, a SiLU base, and two trainable scales initialized to one. A small random spline residual is fitted at initialization. There is no adaptive grid update or additional spline regularizer. The five groups correspond to frequency bands; this grouped activation retains the original linear projection.

The ungrouped model learns six numerator and four denominator coefficients for each of the `5x32` edges in each branch. Its output is `sum_i weight[o,i] * rational[i,o](x[i]) + bias[o]`. No rational coefficients are shared between edges. The initial edge functions match the grouped Swish initialization. These two ablations have model-specific PyTorch implementations in `activations.py`; standard rational configurations call external KAT.

## Optimization

- Adam: learning rate `0.001`, betas `(0.9,0.999)`, epsilon `1e-8`, AMSGrad disabled.
- Weight decay: `0.01` for classifier weight matrices; `0.0001` for other non-activation matrices, including the trainable adjacency; zero for activation coefficients/scales, biases, and batch-normalization parameters.
- Mean cross-entropy loss, no label smoothing.
- 100 epochs, batch size 128, constant learning rate, no early stopping, gradient clipping, or mixed precision.
- One seed (`123` by default) controls Python, NumPy, PyTorch, CUDA, and the shuffled DataLoader. DataLoader workers=0; PyTorch CPU threads=4.
- Incomplete final batches are kept, except a singleton batch is dropped to support batch normalization.
- CuDNN deterministic mode is enabled and benchmarking disabled. These settings do not guarantee bitwise reproducibility of every external CUDA kernel.

## Normalization and checkpoint selection

Every run fits a shared z-score over all subjects and sessions: 15 subjects and three sessions for SEED/SEED-IV; 20 subjects and one session for EEG-Emotion. Statistics are per electrode/frequency feature, with population standard deviation (`ddof=0`). Standard deviations below `1e-6` are replaced by one. Statistics are accumulated in float64 and used with float32 features.

Global normalization includes unlabeled test features. Default `test_peak` selection uses test accuracy and chooses the earliest maximum; `final` selects epoch100 under the default schedule. Test-peak results must be identified as test-selected evaluations. Both modes execute the full configured training schedule.
