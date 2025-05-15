import copy
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

from gr_libs.ml.utils.storage import get_experiment_results_path


def smooth_line(x, y, num_points=300):
    x_smooth = np.linspace(np.min(x), np.max(x), num_points)
    spline = make_interp_spline(x, y, k=3)  # Cubic spline
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


if __name__ == "__main__":

    fragmented_accuracies = {
        "ExpertBasedGraml": {
            "minigrid": {
                "obstacles": {"0.3": [], "0.5": [], "0.7": [], "0.9": [], "1": []},
                "lava_crossing": {"0.3": [], "0.5": [], "0.7": [], "0.9": [], "1": []},
            }
        },
        "Graql": {
            "minigrid": {
                "obstacles": {"0.3": [], "0.5": [], "0.7": [], "0.9": [], "1": []},
                "lava_crossing": {"0.3": [], "0.5": [], "0.7": [], "0.9": [], "1": []},
            }
        },
    }

    continuing_accuracies = copy.deepcopy(fragmented_accuracies)

    # domains = ['panda', 'minigrid', 'point_maze', 'parking']
    domains = ["minigrid"]
    tasks = ["L1", "L2", "L3", "L4", "L5"]
    percentages = ["0.3", "0.5", "1"]

    for partial_obs_type, accuracies, is_same_learn in zip(
        ["fragmented", "continuing"],
        [fragmented_accuracies, continuing_accuracies],
        [False, True],
    ):
        for domain in domains:
            for env in accuracies["ExpertBasedGraml"][domain].keys():
                for task in tasks:
                    graml_res_file_path = f"{get_experiment_results_path(domain, env, task, 'ExpertBasedGraml')}.pkl"
                    graql_res_file_path = (
                        f"{get_experiment_results_path(domain, env, task, 'Graql')}.pkl"
                    )
                    if os.path.exists(graml_res_file_path):
                        with open(graml_res_file_path, "rb") as results_file:
                            results = dill.load(results_file)
                            for percentage in accuracies["expertbasedgraml"][domain][
                                env
                            ].keys():
                                accuracies["expertbasedgraml"][domain][env][
                                    percentage
                                ].append(results[percentage]["accuracy"])
                    else:
                        assert False, f"no file for {graml_res_file_path}"
                    if os.path.exists(graql_res_file_path):
                        with open(graql_res_file_path, "rb") as results_file:
                            results = dill.load(results_file)
                            for percentage in accuracies["expertbasedgraml"][domain][
                                env
                            ].keys():
                                accuracies["Graql"][domain][env][percentage].append(
                                    results[percentage]["accuracy"]
                                )
                    else:
                        assert False, f"no file for {graql_res_file_path}"

    plot_styles = {
        (
            "expertbasedgraml",
            "fragmented",
            0.3,
        ): "g--o",  # Green dashed line with circle markers
        (
            "expertbasedgraml",
            "fragmented",
            0.5,
        ): "g--s",  # Green dashed line with square markers
        (
            "expertbasedgraml",
            "fragmented",
            0.7,
        ): "g--^",  # Green dashed line with triangle-up markers
        (
            "expertbasedgraml",
            "fragmented",
            0.9,
        ): "g--d",  # Green dashed line with diamond markers
        (
            "expertbasedgraml",
            "fragmented",
            1.0,
        ): "g--*",  # Green dashed line with star markers
        (
            "expertbasedgraml",
            "continuing",
            0.3,
        ): "g-o",  # Green solid line with circle markers
        (
            "expertbasedgraml",
            "continuing",
            0.5,
        ): "g-s",  # Green solid line with square markers
        (
            "expertbasedgraml",
            "continuing",
            0.7,
        ): "g-^",  # Green solid line with triangle-up markers
        (
            "expertbasedgraml",
            "continuing",
            0.9,
        ): "g-d",  # Green solid line with diamond markers
        (
            "expertbasedgraml",
            "continuing",
            1.0,
        ): "g-*",  # Green solid line with star markers
        ("Graql", "fragmented", 0.3): "b--o",  # Blue dashed line with circle markers
        ("Graql", "fragmented", 0.5): "b--s",  # Blue dashed line with square markers
        (
            "Graql",
            "fragmented",
            0.7,
        ): "b--^",  # Blue dashed line with triangle-up markers
        ("Graql", "fragmented", 0.9): "b--d",  # Blue dashed line with diamond markers
        ("Graql", "fragmented", 1.0): "b--*",  # Blue dashed line with star markers
        ("Graql", "continuing", 0.3): "b-o",  # Blue solid line with circle markers
        ("Graql", "continuing", 0.5): "b-s",  # Blue solid line with square markers
        ("Graql", "continuing", 0.7): "b-^",  # Blue solid line with triangle-up markers
        ("Graql", "continuing", 0.9): "b-d",  # Blue solid line with diamond markers
        ("Graql", "continuing", 1.0): "b-*",  # Blue solid line with star markers
    }

    def average_accuracies(accuracies, domain):
        avg_acc = {
            algo: {perc: [] for perc in percentages}
            for algo in ["ExpertBasedGraml", "Graql"]
        }

        for algo in avg_acc.keys():
            for perc in percentages:
                for env in accuracies[algo][domain].keys():
                    env_acc = accuracies[algo][domain][env][
                        perc
                    ]  # list of 5, averages for L111 to L555.
                    if env_acc:
                        avg_acc[algo][perc].append(np.array(env_acc))

        for algo in avg_acc.keys():
            for perc in percentages:
                if avg_acc[algo][perc]:
                    avg_acc[algo][perc] = np.mean(np.array(avg_acc[algo][perc]), axis=0)

        return avg_acc

    def plot_domain_accuracies(
        ax,
        fragmented_accuracies,
        continuing_accuracies,
        domain,
        sigma=1,
        line_width=1.5,
    ):
        fragmented_avg_acc = average_accuracies(fragmented_accuracies, domain)
        continuing_avg_acc = average_accuracies(continuing_accuracies, domain)

        x_vals = np.arange(1, 6)  # Number of goals

        # Create "waves" (shaded regions) for each algorithm
        for algo in ["ExpertBasedGraml", "Graql"]:
            fragmented_y_vals_by_percentage = []
            continuing_y_vals_by_percentage = []

            for perc in percentages:
                fragmented_y_vals = np.array(fragmented_avg_acc[algo][perc])
                continuing_y_vals = np.array(continuing_avg_acc[algo][perc])

                # Smooth the trends using Gaussian filtering
                fragmented_y_smoothed = gaussian_filter1d(
                    fragmented_y_vals, sigma=sigma
                )
                continuing_y_smoothed = gaussian_filter1d(
                    continuing_y_vals, sigma=sigma
                )

                fragmented_y_vals_by_percentage.append(fragmented_y_smoothed)
                continuing_y_vals_by_percentage.append(continuing_y_smoothed)

                ax.plot(
                    x_vals,
                    fragmented_y_smoothed,
                    plot_styles[(algo, "fragmented", float(perc))],
                    label=f"{algo}, non-consecutive, {perc}",
                    linewidth=0.5,  # Control line thickness here
                )
                ax.plot(
                    x_vals,
                    continuing_y_smoothed,
                    plot_styles[(algo, "continuing", float(perc))],
                    label=f"{algo}, consecutive, {perc}",
                    linewidth=0.5,  # Control line thickness here
                )

        ax.set_xticks(x_vals)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylim([0, 1])
        ax.set_title(f"{domain.capitalize()} Domain", fontsize=16)
        ax.grid(True)

    fig, axes = plt.subplots(
        1, 4, figsize=(24, 6)
    )  # Increase the figure size for better spacing (width 24, height 6)

    # Generate each plot in a subplot, including both fragmented and continuing accuracies
    for i, domain in enumerate(domains):
        plot_domain_accuracies(
            axes[i], fragmented_accuracies, continuing_accuracies, domain
        )

    # Set a single x-axis and y-axis label for the entire figure
    fig.text(
        0.5, 0.04, "Number of Goals", ha="center", fontsize=20
    )  # Centered x-axis label
    fig.text(
        0.04, 0.5, "Accuracy", va="center", rotation="vertical", fontsize=20
    )  # Reduced spacing for y-axis label

    # Adjust subplot layout to avoid overlap
    plt.subplots_adjust(
        left=0.09, right=0.91, top=0.79, bottom=0.21, wspace=0.3
    )  # More space on top (top=0.82)

    # Place the legend above the plots with more space between legend and plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=12,
    )  # Moved above with bbox_to_anchor

    # Save the figure and show it
    save_dir = os.path.join("figures", "all_domains_accuracy_plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "accuracy_plots_smooth.png"), dpi=300)
