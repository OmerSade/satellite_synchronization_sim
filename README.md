# Satellite Synchronization Simulation

This repository contains a clean Python simulation framework for comparing an
inter-satellite distributed time synchronization (IS-DTS) algorithm with a simple
Precision Time Protocol (PTP)-like baseline in a LEO satellite network.

The model is inspired by Han et al., **"Inter-satellite distributed time
synchronization solution with nanosecond accuracy in satellite networks"**,
*Optics Express* 33(7), 14555-14565 (2025), DOI: `10.1364/OE.543159`. The paper
proposes distributed, parallel inter-satellite time exchange to reduce the impact
of dynamic link latency and demonstrates robustness under topology changes and
node failures. This project implements a compact educational simulation rather
than a bit-accurate reproduction of the paper.

## Project structure

```text
src/
  satellite.py        # Satellite node state and clock model
  network.py          # Graph topology and asymmetric link-delay model
  isdts.py            # Distributed parallel IS-DTS-style algorithm
  ptp.py              # Simplified serial master-slave PTP baseline
  simulation.py       # Main comparison simulation loop
utils/
  metrics.py          # Error and convergence metrics
  plotting.py         # Matplotlib visualization helpers
experiments/
  run_simulation.py   # Command-line experiment runner
requirements.txt
README.md
```

## Simulation model

- The network contains a configurable number of satellites, defaulting to 72.
- Each satellite has:
  - a local clock offset in nanoseconds,
  - a small oscillator drift in nanoseconds per second,
  - an operational state: `Initializing`, `Listening`, `Adjusting`, or `Faulty`.
- The topology is a graph-based LEO-like mesh, not a full orbital propagator.
  Ring links model nearby same-plane neighbors, and random cross-links approximate
  inter-plane optical links.
- Time advances in discrete simulation steps.
- Optional link toggling and node-failure events can be enabled from the CLI.

## IS-DTS-style algorithm

The IS-DTS implementation is distributed and parallel:

1. All healthy satellites exchange timestamp information with active neighbors in
   the same simulation step.
2. Each active link generates a pairwise offset estimate between neighboring
   clocks. Residual measurement error includes timestamp noise and directed-delay
   asymmetry.
3. Each satellite averages offset estimates from healthy neighbors.
4. The local clock is corrected toward that neighborhood average.
5. Faulty satellites are ignored by healthy nodes.

This produces a consensus-like synchronization process with no permanent master
node.

## Baseline PTP-like algorithm

The baseline uses a serial master-slave approach:

1. Satellite `0` acts as the master clock.
2. Each healthy non-master satellite estimates its offset to the master.
3. The estimate assumes symmetric propagation delay.
4. Directed link-delay asymmetry leaves a residual bias, demonstrating why serial
   timestamp exchange can be less accurate in dynamic satellite networks.

## Metrics and plots

The experiment records:

- peak-to-peak time difference across healthy satellites,
- RMS error after subtracting the healthy-node mean,
- time to convergence below a configurable threshold.

The default runner writes a comparison plot to `outputs/error_comparison.png`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_simulation.py
```

Run without generating a plot:

```bash
python experiments/run_simulation.py --no-plots
```

Example with topology changes and a node failure:

```bash
python experiments/run_simulation.py \
  --nodes 72 \
  --steps 150 \
  --link-toggle-probability 0.03 \
  --fail-node 50:12
```

## Notes and next steps

This first version focuses on the core project structure and a working simulation.
Useful extensions include:

- replacing the graph mesh with orbital-plane geometry and polar-region link
  rules,
- adding hardware timestamp quantization and clock-noise models,
- implementing weighted neighbor selection based on link quality,
- exporting metrics to CSV for batch experiments,
- adding unit tests for convergence and fault handling.
