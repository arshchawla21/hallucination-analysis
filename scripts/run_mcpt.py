"""Token-shuffle MCPT on a saved config.

    python scripts/run_mcpt.py configs/spike_<stamp>.json [--sample in|out|both] [-n 200]

Small p  => the signal exploits real TEMPORAL structure (order matters).
Large p  => its edge is explained by the marginal uncertainty distribution
            alone -- an order-free baseline would do just as well.

Out-sample is fast (one backtest per permutation); in-sample refits the
whole search space per permutation and is proportionally slower. Plot +
JSON land in reports/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcpt.mcpt import plot_mcpt, run_mcpt, save_mcpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="configs/<signal>_<stamp>.json")
    ap.add_argument("--sample", choices=("in", "out", "both"), default="out")
    ap.add_argument("-n", "--n-perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    Path("reports").mkdir(exist_ok=True)
    samples = ("in", "out") if args.sample == "both" else (args.sample,)
    stem = Path(args.config).stem

    for sample in samples:
        # in-sample refits per permutation -- keep it affordable by default
        n = args.n_perms if sample == "out" else min(args.n_perms, 50)
        print(f"\n=== {sample}-sample MCPT ({n} permutations) ===")
        result = run_mcpt(args.config, sample=sample, n_perms=n,
                          seed=args.seed, data_path=args.data)
        png = Path("reports") / f"mcpt_{sample}_{stem}.png"
        js = Path("reports") / f"mcpt_{sample}_{stem}.json"
        plot_mcpt(result, png)
        save_mcpt(result, js)
        print(f"p = {result['p_value']:.4f}  "
              f"(real {result['objective']} = {result['real_value']:.4f})")
        print(f"wrote {png} and {js}")


if __name__ == "__main__":
    main()
