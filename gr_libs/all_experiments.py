""" executes odgr_executor parallely on a set of problems defined in consts.py """

import argparse
import concurrent.futures
import os
import subprocess
import sys

import dill
import numpy as np

from gr_libs.ml.utils.storage import get_experiment_results_path

parser = argparse.ArgumentParser()
parser.add_argument("--domains", nargs="+", required=True, help="List of domains")
parser.add_argument(
    "--envs",
    nargs="+",
    required=True,
    help="List of environments (same order as domains)",
)
parser.add_argument(
    "--tasks", nargs="+", required=True, help="List of tasks (e.g. L1 L2 L3 L4 L5)"
)
parser.add_argument(
    "--recognizers", nargs="+", required=True, help="List of recognizers"
)
parser.add_argument(
    "--n", type=int, default=5, help="Number of times to execute each task"
)
args = parser.parse_args()

# Build configs dynamically
configs = {}
for domain, env in zip(args.domains, args.envs):
    configs.setdefault(domain, {})
    configs[domain][env] = args.tasks

recognizers = args.recognizers
n = args.n


# Function to read results from the result file
def read_results(res_file_path):
    """
    Read the results from a result file.

    Args:
        res_file_path (str): The path to the result file.

    Returns:
        The results read from the file.
    """
    with open(res_file_path, "rb") as f:
        results = dill.load(f)
    return results


# Every thread worker executes this function.
def run_experiment(domain, env, task, recognizer, i, generate_new=False):
    """
    Run an experiment.

    Args:
        domain (str): The domain of the experiment.
        env (str): The environment of the experiment.
        task (str): The task of the experiment.
        recognizer (str): The recognizer used in the experiment.
        i (int): The index of the experiment.
        generate_new (bool, optional): Whether to generate new results.
        Defaults to False.

    Returns:
        tuple: A tuple containing the experiment details and the results.
    """
    cmd = f"python gr_libs/odgr_executor.py --domain {domain} --recognizer {recognizer} --env_name {env} --task {task} --collect_stats --experiment_num {i}"
    try:
        res_file_path = get_experiment_results_path(domain, env, task, recognizer)
        i_res_file_path_pkl = os.path.join(res_file_path, f"res_{i}.pkl")
        i_res_file_path_txt = os.path.join(res_file_path, f"res_{i}.txt")
        if generate_new or (
            not os.path.exists(i_res_file_path_txt)
            or not os.path.exists(i_res_file_path_pkl)
        ):
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"Execution failed: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
                return None
            else:
                print(f"Finished execution successfully: {cmd}")
        else:
            print(
                f"File {i_res_file_path_txt} already exists. Skipping execution of {cmd}"
            )
        return ((domain, env, task, recognizer), read_results(i_res_file_path_pkl))
    except Exception as e:
        print(f"Exception occurred while running experiment: {e}")
        return None


# Collect results
results = {}

# create an executor that manages a pool of threads.
# Note that any failure in the threads will not stop the main thread
# from continuing and vice versa, nor will the debugger view the
# failure if in debug mode.
# Use prints and if any thread's printing stops suspect failure.
# If failure happened, use breakpoints before failure and use the
# watch to see the failure by pasting the problematic piece of code.
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for domain, envs in configs.items():
        for env, tasks in envs.items():
            for task in tasks:
                for recognizer in recognizers:
                    for i in range(n):
                        futures.append(
                            executor.submit(
                                run_experiment,
                                domain,
                                env,
                                task,
                                recognizer,
                                i,
                                generate_new=(
                                    True
                                    if len(sys.argv) > 1
                                    and sys.argv[1] == "--generate_new"
                                    else False
                                ),
                            )
                        )

    for future in concurrent.futures.as_completed(futures):
        if future.result() is None:
            print(
                f"for future {future}, future.result() is None. \
                 Continuing to next future."
            )
            continue
        key, result = future.result()
        print(f"main thread reading results from future {key}")
        if key not in results:
            results[key] = []
        results[key].append(result)

# Calculate average accuracy and standard deviation for each percentage
detailed_summary = {}
compiled_accuracies = {}
for key, result_list in results.items():
    domain, env, task, recognizer = key
    percentages = result_list[0].keys()
    detailed_summary[key] = {}
    if (domain, recognizer) not in compiled_accuracies:
        compiled_accuracies[(domain, recognizer)] = {}
    for percentage in percentages:
        if percentage == "total":
            continue
        if percentage not in compiled_accuracies[(domain, recognizer)].keys():
            compiled_accuracies[(domain, recognizer)][percentage] = {}
        if percentage not in detailed_summary[key].keys():
            detailed_summary[key][percentage] = {}
        consecutive_accuracies = [
            result[percentage]["consecutive"]["accuracy"] for result in result_list
        ]
        non_consecutive_accuracies = [
            result[percentage]["non_consecutive"]["accuracy"] for result in result_list
        ]
        if (
            "consecutive"
            in compiled_accuracies[(domain, recognizer)][percentage].keys()
        ):
            compiled_accuracies[(domain, recognizer)][percentage]["consecutive"].extend(
                consecutive_accuracies
            )
        else:
            compiled_accuracies[(domain, recognizer)][percentage][
                "consecutive"
            ] = consecutive_accuracies
        if (
            "non_consecutive"
            in compiled_accuracies[(domain, recognizer)][percentage].keys()
        ):
            compiled_accuracies[(domain, recognizer)][percentage][
                "non_consecutive"
            ].extend(non_consecutive_accuracies)
        else:
            compiled_accuracies[(domain, recognizer)][percentage][
                "non_consecutive"
            ] = non_consecutive_accuracies
        avg_consecutive_accuracy = np.mean(consecutive_accuracies)
        consecutive_std_dev = np.std(consecutive_accuracies)
        detailed_summary[key][percentage]["consecutive"] = (
            avg_consecutive_accuracy,
            consecutive_std_dev,
        )
        avg_non_consecutive_accuracy = np.mean(non_consecutive_accuracies)
        non_consecutive_std_dev = np.std(non_consecutive_accuracies)
        detailed_summary[key][percentage]["non_consecutive"] = (
            avg_non_consecutive_accuracy,
            non_consecutive_std_dev,
        )

compiled_summary = {}
for key, percentage_dict in compiled_accuracies.items():
    compiled_summary[key] = {}
    for percentage, cons_accuracies in percentage_dict.items():
        compiled_summary[key][percentage] = {}
        for is_cons, accuracies in cons_accuracies.items():
            avg_accuracy = np.mean(accuracies)
            std_dev = np.std(accuracies)
            compiled_summary[key][percentage][is_cons] = (avg_accuracy, std_dev)

# Write different summary results to different files, one per recognizer
if not os.path.exists(os.path.join("outputs", "summaries")):
    os.makedirs(os.path.join("outputs", "summaries"))

for recognizer in recognizers:
    compiled_summary_file_path = os.path.join(
        "outputs",
        "summaries",
        f"compiled_summary_{''.join(configs.keys())}_{recognizer}.txt",
    )
    with open(compiled_summary_file_path, "w") as f:
        for key, percentage_dict in compiled_summary.items():
            domain, recog = key
            if recog != recognizer:
                continue  # Only write results for this recognizer
            for percentage, cons_info in percentage_dict.items():
                for is_cons, (avg_accuracy, std_dev) in cons_info.items():
                    f.write(
                        f"{domain}\t{recog}\t{percentage}\t{is_cons}\t{avg_accuracy:.4f}\t{std_dev:.4f}\n"
                    )
    print(f"Compiled summary results written to {compiled_summary_file_path}")

    detailed_summary_file_path = os.path.join(
        "outputs",
        "summaries",
        f"detailed_summary_{''.join(configs.keys())}_{recognizer}.txt",
    )
    with open(detailed_summary_file_path, "w") as f:
        for key, percentage_dict in detailed_summary.items():
            domain, env, task, recog = key
            if recog != recognizer:
                continue  # Only write results for this recognizer
            f.write(f"{domain}\t{env}\t{task}\t{recog}\n")
            for percentage, cons_info in percentage_dict.items():
                for is_cons, (avg_accuracy, std_dev) in cons_info.items():
                    f.write(
                        f"\t\t{percentage}\t{is_cons}\t{avg_accuracy:.4f}\t{std_dev:.4f}\n"
                    )
    print(f"Detailed summary results written to {detailed_summary_file_path}")
