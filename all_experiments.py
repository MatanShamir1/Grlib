import dill
import subprocess
import concurrent.futures
import numpy as np

from grlib.ml.utils.storage import get_experiment_results_path, set_global_storage_configs

# Define the lists
# domains = ['minigrid', 'point_maze', 'parking', 'panda']
# envs = {
#     'minigrid': ['obstacles', 'lava_crossing'],
#     'point_maze': ['four_rooms', 'lava_crossing'],
#     'parking': ['gc_agent', 'gd_agent'],
#     'panda': ['gc_agent', 'gd_agent']
# }
# tasks = {
#     'minigrid': ['L111', 'L222', 'L333', 'L444', 'L555'],
#     'point_maze': ['L111', 'L222', 'L333', 'L444', 'L555'],
#     'parking': ['L111', 'L222', 'L333', 'L444', 'L555'],
#     'panda': ['L111', 'L222', 'L333', 'L444', 'L555']
# }
domains = ['panda']
envs = {
    'panda': ['gc_agent']
}
tasks = {
    'panda': ['L111', 'L222', 'L333', 'L444', 'L555']
}
partial_obs_types = ['fragmented', 'continuing']
recognizers = ['graml', 'graql']
n = 5  # Number of times to execute each task
percentages = ['0.3', '0.5', '0.7', '0.9', '1']

# Every thread worker executes this function.
def run_experiment(domain, env, task, partial_obs_type, recognizer):
    cmd = f"python experiments.py --domain {domain} --{domain}_env {env} --task {task} --partial_obs_type {partial_obs_type} --recognizer {recognizer} --collect_stats"
    global_configs = {'recognizer_str': recognizer, 'is_fragmented': partial_obs_type}
    if recognizer == 'graml':
        cmd += " --inference_same_seq_len"
        if partial_obs_type == 'continuing':
            cmd += " --learn_same_seq_len"
        global_configs['is_inference_same_length_sequences'] = True
        global_configs['is_learn_same_length_sequences'] = partial_obs_type == 'continuing'
    print(f"Starting execution: {cmd}")
    try:
        # every thread in the current process starts a new process which executes the command. the current thread waits for the subprocess to finish.
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Execution failed: {cmd}")
            print(f"Error: {result.stderr}")
            return None
        else:
            print(f"Finished execution successfully: {cmd}")
        set_global_storage_configs(**global_configs)
        res_file_path = get_experiment_results_path(domain, env, task)
        return ((domain, env, task, partial_obs_type, recognizer), res_file_path)
    except Exception as e:
        print(f"Exception occurred while running experiment: {e}")
        return None

# Function to read results from the result file
def read_results(res_file_path):
    with open(res_file_path, 'rb') as f:
        results = dill.load(f)
    return results

# Collect results
results = {}

# create an executor that manages a pool of threads.
# Note that any failure in the threads will not stop the main thread from continuing and vice versa, nor will the debugger view the failure if in debug mode.
# Use prints and if any thread's printing stops suspect failure. If failure happened, use breakpoints before failure and use the watch to see the failure by pasting the problematic piece of code.
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for domain in domains:
        for env in envs[domain]:
            for task in tasks[domain]:
                for partial_obs_type in partial_obs_types:
                    for recognizer in recognizers:
                        for i in range(n):
                            # submit returns a future object that represents the execution of the function by some thread in the pool. the method is added to the list and the executor will run it as soon as it can.
                            futures.append(executor.submit(run_experiment, domain, env, task, partial_obs_type, recognizer))

    # main thread continues execution after the loop above. Here, it waits for all threads. every time a thread finishes, the main thread reads the results from the future object.
    # If debugging main thread, note the executor will stop creating new threads and running them since it exists and runs in the main thread.
    # probably, the main thread gets interrupted by the return of a future and knows to start execution of new threads and then continue main thread execution
    for future in concurrent.futures.as_completed(futures):
        # the objects returned by the 'result' func are tuples with key being the args inserted to 'submit'.
        if future.result() is None:
            print(f"for future {future}, future.result() is None. Continuing to next future.")
            continue
        key, res_file_path = future.result()
        print(f"main thread reading results from future {key}")
        result = read_results(f"{res_file_path}.pkl")
        # list because every experiment is executed n times.
        if key not in results:
            results[key] = []
        results[key].append(result)

# Calculate average accuracy and standard deviation for each percentage
detailed_summary = {}
compiled_accuracies = {}
for key, result_list in results.items():
    domain, env, task, partial_obs_type, recognizer = key
    percentages = result_list[0].keys()
    detailed_summary[key] = {}
    if (domain, partial_obs_type, recognizer) not in compiled_accuracies:
        compiled_accuracies[(domain, partial_obs_type, recognizer)] = {}
    for percentage in percentages:
        if percentage == 'total':
            continue
        accuracies = [result[percentage]['accuracy'] for result in result_list] # accuracies in all different n executions
        if percentage in compiled_accuracies[(domain, partial_obs_type, recognizer)]:
            compiled_accuracies[(domain, partial_obs_type, recognizer)][percentage].extend(accuracies)
        else:
            compiled_accuracies[(domain, partial_obs_type, recognizer)][percentage] = accuracies
        avg_accuracy = np.mean(accuracies)
        std_dev = np.std(accuracies)
        detailed_summary[key][percentage] = (avg_accuracy, std_dev)
     
compiled_summary = {}   
for key, percentage_dict in compiled_accuracies.items():
    compiled_summary[key] = {}
    for percentage, accuracies in percentage_dict.items():
        avg_accuracy = np.mean(accuracies)
        std_dev = np.std(accuracies)
        compiled_summary[key][percentage] = (avg_accuracy, std_dev)

# Write different summary results to different files
detailed_summary_file_path = f"detailed_summary_{''.join(domains)}.txt"
compiled_summary_file_path = f"compiled_summary_{''.join(domains)}.txt"
with open(detailed_summary_file_path, 'w') as f:
    for key, percentage_dict in detailed_summary.items():
        domain, env, task, partial_obs_type, recognizer = key
        f.write(f"{domain}\t{env}\t{task}\t{partial_obs_type}\t{recognizer}\n")
        for percentage, (avg_accuracy, std_dev) in percentage_dict.items():
            f.write(f"\t\t{percentage}\t{avg_accuracy:.4f}\t{std_dev:.4f}\n")
            
with open(compiled_summary_file_path, 'w') as f:
    for key, percentage_dict in compiled_summary.items():
        domain, partial_obs_type, recognizer = key
        f.write(f"{domain}\t{partial_obs_type}\t{recognizer}\n")
        for percentage, (avg_accuracy, std_dev) in percentage_dict.items():
            f.write(f"\t{percentage}\t{avg_accuracy:.4f}\t{std_dev:.4f}\n")

print(f"Detailed summary results written to {detailed_summary_file_path}")
print(f"Compiled summary results written to {compiled_summary_file_path}")