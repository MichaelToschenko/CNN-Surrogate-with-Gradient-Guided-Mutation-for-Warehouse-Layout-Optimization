**English** | [Русский](README_RU.md)

# CNN Surrogate with Gradient-Guided Mutation for Warehouse Layout Optimization

This project addresses the problem of optimal placement of entry points, exits, and storage zones on the rectangular grid of an automated warehouse. The quality of a configuration is measured by the number of discrete-event simulation steps required to process a fixed number of containers by mobile robots. Two algorithms are implemented: a baseline genetic algorithm (**GA**) and its modification (**CNN-GA**), in which a convolutional neural network acts as a surrogate for the fitness function - both for candidate selection and as a source of gradient for guided mutation.

## Optimization process

![Layout evolution](plots/evolution_cnn_ga.png)

Evolution of the best layout over generations (CNN-GA, 15x15 grid).

## Problem formulation

- **Warehouse representation.** A rectangular mxn grid, each cell of one of 4 types: `Road`, `Entry`, `Exit`, `Save`. The number of cells of each type is given as a parameter and preserved throughout optimization.
- **Simulation.** At each step, a container appears at each `Entry` cell with probability `p`. A dispatcher assigns tasks to robots by a nearest-free-robot rule; routes are computed by breadth-first search over the `Road`-cell network. A container follows the path `Entry -> Save -> Exit`.
- **Objective function.** `T(x)` - the number of simulation steps required to process a given number of containers under configuration `x`. The task is to minimize `T(x)`.

## Method

**GA (baseline).** Tournament selection, single-point horizontal/vertical crossover, relocation mutation (moving a non-Road cell to a random Road position), elitism.

**CNN-GA.** The same operators, plus two mechanisms based on a trained CNN surrogate `f_hat_theta(X)`:

1. **Pre-screening.** `alpha*(N-E)` candidates are generated; the surrogate predicts their fitness, and only the `N-E` best ones are passed to the simulation.
2. **Gradient-guided mutation.** The gradient `grad_X f_hat_theta(X)` ranks possible cell relocations; the top-T pairs are verified by a full surrogate forward pass, and the best one is applied.

The first `n_a` generations form the warm-up phase where training data is collected (the surrogate is not used). After that the surrogate is fine-tuned every `r` generations. **The simulation evaluation budget is identical for GA and CNN-GA** - the comparison is fair.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Two entry points (from the project root):

```bash
python -m experiments.comparison      # GA vs CNN-GA, 5 seeds, all plots saved to plots/
python -m experiments.evolution_viz   # Strip of best-layout snapshots over generations
```

Both scripts reuse cached logs from `logs/`. To force a fresh run, delete the corresponding `logs/run_*.json` file.

Interactive example:

```python
from algorithms import GeneticAlgorithm, CNNGuidedGA

ga = GeneticAlgorithm(
    m=15, n=15, entries=5, exits=5, saves=30,
    num_robots=12, containers_to_process=160, prob=0.35,
    pop_size=50, generations=80, n_jobs=-1,
)
best_config, log = ga.run()
best_config.plot()
```

## Results

**Experiment conditions:** 15x15 grid, 5 entries / 5 exits / 30 storage cells, 12 robots, 160 containers, `p=0.35`. Evolution parameters: `N=50`, `G=80`. 5 runs with seeds `{42, 123, 456, 789, 1337}`.

### Convergence

![Best-fitness convergence](plots/convergence.png)

Best fitness per generation (mean +/- 95% CI, 5 runs). CNN-GA consistently outperforms GA after the surrogate is activated at generation 5; the confidence intervals of the two algorithms barely overlap across the plateau.

### Generations to reach a threshold

Number of generations at which the algorithm first reached the given fitness threshold (mean over runs that reached it; in parentheses - fraction of such runs out of 5).

| Threshold | GA        | CNN-GA    |
|-----------|-----------|-----------|
| <=210      | 2 (5/5)   | 2 (5/5)   |
| <=200      | 3 (5/5)   | 4 (5/5)   |
| <=190      | 6 (5/5)   | 5 (5/5)   |
| <=180      | 7 (5/5)   | 6 (5/5)   |
| <=170      | 14 (5/5)  | 10 (5/5)  |
| <=165      | 18 (5/5)  | 11 (5/5)  |
| <=160      | 20 (3/5)  | 12 (5/5)  |
| <=155      | 24 (2/5)  | 16 (5/5)  |
| <=150      | 36 (1/5)  | 23 (3/5)  |
| <=145      | - (0/5)   | 35 (3/5)  |
| <=140      | - (0/5)   | 34 (1/5)  |

On loose thresholds the difference is small; starting from <=170 CNN-GA systematically reaches each level 3-12 generations earlier than GA, and the tight thresholds <=145 and <=140 are not reached by GA in any run.

### Mean population fitness

![Mean population fitness](plots/mean_fitness.png)

The gap between the algorithms widens immediately after surrogate activation and persists until the end of the runs - the effect of CNN-GA is visible not only in the elite but also in overall population quality.

### Final fitness

![Final-fitness distribution](plots/final_fitness_boxplot.png)

Distribution of final values across 5 runs. GA median - 157, CNN-GA median - 144. Paired t-test: `t=6.076`, `p=0.0037`, `df=4` - a statistically significant ~8% advantage of CNN-GA.

### Surrogate quality

![Surrogate R^2](plots/cnn_surrogate_r2.png)

Coefficient of determination `R^2` of the surrogate across fine-tuning cycles. After a few early high-variance cycles it levels off at ~0.95 - predictions are close to the true values, which explains why the gradient signal is informative for the mutation.

## Project structure

```
.
|-- algorithms/       # GA, CNN-GA, surrogate, mutation operators
|-- experiments/      # comparison.py, evolution_viz.py, cases.py, plots.py
|-- simulation.py     # discrete-event warehouse simulator
|-- individual.py     # configuration genome
|-- metrics.py        # spatial metrics
|-- constants.py      # cell-type and state enums
|-- plots/            # generated plots (png)
|-- logs/             # cached run logs (json)
\-- requirements.txt
```
