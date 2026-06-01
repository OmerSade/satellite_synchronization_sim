# Satellite Synchronization Simulation

Educational Python simulation for distributed time synchronization in a low Earth
orbit (LEO) satellite mesh. The simulator compares nanosecond-scale consensus
behavior across inter-satellite optical and RF timestamp exchanges, records run
metrics, and can repeat the experiment with Monte Carlo parameter draws.

The model is inspired by Han et al., **"Inter-satellite distributed time
synchronization solution with nanosecond accuracy in satellite networks"**,
*Optics Express* 33(7), 14555-14565 (2025), DOI: `10.1364/OE.543159`. This
repository is a compact educational implementation rather than a bit-accurate
reproduction of the paper.

## Repository layout

```text
simulation.py                     # Main CLI, satellite clock model, topology, metrics, outputs
satellite_laser_communication.py  # Optical link budget, acquisition, timing-noise model
satellite_receiver.py             # RF link budget, frame reception, BER/timestamp model
plots.py                          # Matplotlib plots for regular and Monte Carlo runs
parameters                        # JSON configuration used by default
requirements.txt                  # Runtime Python dependencies
outputs/                          # Generated run folders (created at runtime, git-ignored)
```

> Note: `simulation.py` installs a small compatibility module named
> `src.satellite` at runtime because the standalone RF and laser communication
> modules import `Satellite` and `SatelliteState` through that legacy path.

## What the simulation models

- A configurable constellation size (`simulation.satellite_count`) with a
  ring-plus-random-crosslink topology.
- Per-satellite clock offset and drift in nanoseconds.
- Healthy and faulty satellite states, including optional failure events during a
  run.
- Parallel neighbor exchanges on active links each synchronization round.
- Link dropout probability to emulate unavailable inter-satellite links.
- Weighted corrections from two measurement channels:
  - laser optical downlink measurements, controlled by `laser_weight`, and
  - RF frame timing measurements, controlled by `rf_weight`.
- Consensus-style clock correction using `distributed_gain`.

## Configuration

All tunable values live in the root-level `parameters` JSON file. Important
sections are:

- `simulation`: constellation size, number of steps, random seed, clock-error
  distributions, correction gains, dropout probability, faulty nodes, and failure
  events.
- `geometry`: nominal link range, range jitter, elevation, pointing error,
  atmospheric transmittance, and relative velocity spread.
- `laser`: optical wavelength, power, aperture sizes, detector/background noise,
  acquisition threshold, and timing jitter floor.
- `rf`: carrier frequency, sample/symbol rates, power/gains/losses, noise model,
  synchronization threshold, phase noise, and frame size.
- `monte_carlo`: optional repeated-run settings and parameter distributions.
- `outputs`: output root directory and plot generation toggle.

To try a custom configuration, copy `parameters`, edit the copy, and pass it to
the CLI with `--parameters`.

## Outputs

Each run creates a timestamped folder under `outputs/` by default. A regular run
contains:

- `parameters.json`: exact configuration snapshot used for the run.
- `metrics.csv`: per-step peak-to-peak error, RMS error, mean laser SNR, mean RF
  Eb/No, and accepted measurement count.
- `clock_offsets.csv`: per-step offset for every satellite.
- `summary.json`: final RMS error, final peak-to-peak error, convergence step,
  and accepted-measurement total.
- `error_evolution.png`, `clock_offsets.png`, and `link_quality.png` when plots
  are enabled.

A Monte Carlo run creates one parent output folder with one `draw_XXX/` child
folder per draw, plus `monte_carlo_summary.json` and aggregate Monte Carlo plots.

## Quick start

The project requires Python 3.10 or newer because it uses modern type-hint
syntax such as `str | Path`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python simulation.py
```

Run a single simulation even if the parameter file enables Monte Carlo mode:

```bash
python simulation.py --single-run
```

Run Monte Carlo mode from the existing parameter file:

```bash
python simulation.py --monte-carlo
```

Run with a custom parameter file:

```bash
python simulation.py --parameters path/to/parameters.json
```

Disable plot generation by setting `outputs.save_plots` to `false` in the
parameter file. CSV and JSON summaries are still written for downstream analysis.

## Generated metrics

The main convergence metrics are:

- **Peak-to-peak error**: max minus min clock offset across healthy satellites.
- **RMS error**: RMS of healthy-satellite offsets after subtracting the healthy
  mean offset.
- **Convergence step**: first step where RMS error is at or below
  `simulation.convergence_threshold_ns`; `null` means the run did not converge
  within the configured step count.
- **Accepted measurements**: number of laser/RF measurements that passed their
  acquisition or synchronization thresholds in a step.

## Development notes

- The repository is intentionally lightweight and does not currently include a
  package build system or automated test suite.
- `requirements.txt` lists only third-party runtime dependencies; the remaining
  imports are from the Python standard library.
- Generated output folders can become large during Monte Carlo runs and should
  remain untracked.
- Useful next improvements include unit tests for convergence/failure handling,
  optional CSV gating via `outputs.save_csv`, richer orbital-plane geometry, and
  batch-analysis notebooks.
