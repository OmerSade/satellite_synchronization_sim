"""Command-line entry point for satellite synchronization experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import SimulationConfig, SynchronizationSimulation
from utils.plotting import plot_error_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IS-DTS with a PTP baseline.")
    parser.add_argument("--nodes", type=int, default=72, help="Number of satellites.")
    parser.add_argument("--steps", type=int, default=120, help="Simulation steps.")
    parser.add_argument("--dt", type=float, default=1.0, help="Step duration in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--link-toggle-probability",
        type=float,
        default=0.0,
        help="Probability that each link is disconnected on a step.",
    )
    parser.add_argument(
        "--fail-node",
        action="append",
        default=[],
        metavar="STEP:NODE",
        help="Mark NODE faulty at STEP. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Run metrics only; do not write figures."
    )
    parser.add_argument(
        "--output",
        default="outputs/error_comparison.png",
        help="Output path for the comparison plot.",
    )
    return parser.parse_args()


def parse_failures(items: list[str]) -> dict[int, list[int]]:
    failures: dict[int, list[int]] = {}
    for item in items:
        step_text, node_text = item.split(":", maxsplit=1)
        failures.setdefault(int(step_text), []).append(int(node_text))
    return failures


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        num_satellites=args.nodes,
        steps=args.steps,
        dt_s=args.dt,
        seed=args.seed,
        link_toggle_probability=args.link_toggle_probability,
        failure_events=parse_failures(args.fail_node),
    )
    simulation = SynchronizationSimulation(config)
    result = simulation.run()

    print("Simulation complete")
    print(f"Final IS-DTS max error: {result.isdts_max_error_ns[-1]:.3f} ns")
    print(f"Final PTP max error:    {result.ptp_max_error_ns[-1]:.3f} ns")
    print(f"IS-DTS convergence:     {result.isdts_convergence_s} s")
    print(f"PTP convergence:        {result.ptp_convergence_s} s")

    if not args.no_plots:
        output = plot_error_comparison(
            result.times_s,
            result.isdts_max_error_ns,
            result.ptp_max_error_ns,
            args.output,
        )
        print(f"Plot written to {output}")


if __name__ == "__main__":
    main()
