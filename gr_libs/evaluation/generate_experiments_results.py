import argparse
import os

import dill
import matplotlib.pyplot as plt
import numpy as np

from gr_libs.ml.utils.storage import get_experiment_results_path


def load_results(domain, env, task, recognizer, n_runs, percentage, cons_type):
    # Collect accuracy for a single task and recognizer
    accs = []
    res_dir = get_experiment_results_path(domain, env, task, recognizer)
    if not os.path.exists(res_dir):
        return accs
    for i in range(n_runs):
        res_file = os.path.join(res_dir, f"res_{i}.pkl")
        if not os.path.exists(res_file):
            continue
        with open(res_file, "rb") as f:
            results = dill.load(f)
        if percentage in results and cons_type in results[percentage]:
            acc = results[percentage][cons_type].get("accuracy")
            if acc is not None:
                accs.append(acc)
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--recognizers", nargs="+", required=True)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--percentage", required=True)
    parser.add_argument(
        "--cons_type", choices=["consecutive", "non_consecutive"], required=True
    )
    parser.add_argument("--graph_name", type=str, default="experiment_results")
    args = parser.parse_args()

    plt.figure(figsize=(7, 5))
    has_data = False
    missing_recognizers = []

    for recognizer in args.recognizers:
        x_vals = []
        y_means = []
        y_sems = []
        for task in args.tasks:
            accs = load_results(
                args.domain,
                args.env,
                task,
                recognizer,
                args.n_runs,
                args.percentage,
                args.cons_type,
            )
            if accs:
                x_vals.append(task)
                y_means.append(np.mean(accs))
                y_sems.append(np.std(accs) / np.sqrt(len(accs)))
        if x_vals:
            has_data = True
            x_ticks = np.arange(len(x_vals))
            plt.plot(x_ticks, y_means, marker="o", label=recognizer)
            plt.fill_between(
                x_ticks,
                np.array(y_means) - np.array(y_sems),
                np.array(y_means) + np.array(y_sems),
                alpha=0.2,
            )
            plt.xticks(x_ticks, x_vals)
        else:
            print(
                f"Warning: No data found for recognizer '{recognizer}' in {args.domain} / {args.env} / {args.percentage} / {args.cons_type}"
            )
            missing_recognizers.append(recognizer)

    if not has_data:
        raise RuntimeError(
            f"No data found for any recognizer in {args.domain} / {args.env} / {args.percentage} / {args.cons_type}. "
            f"Missing recognizers: {', '.join(missing_recognizers)}"
        )

    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title(f"{args.domain} - {args.env} ({args.percentage}, {args.cons_type})")
    plt.legend()
    plt.grid(True)
    fig_path = f"{args.graph_name}_{'_'.join(args.recognizers)}_{args.domain}_{args.env}_{args.percentage}_{args.cons_type}.png"
    plt.savefig(fig_path)
    print(f"Figure saved at: {fig_path}")


if __name__ == "__main__":
    main()
