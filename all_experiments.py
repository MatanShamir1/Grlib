import os
import sys
import threading
import dill
import subprocess
import concurrent.futures
import numpy as np

from gr_libs.ml.utils.storage import get_experiment_results_path

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
configs = {
    'minigrid': {
        'MiniGrid-SimpleCrossingS13N4': ['L1', 'L2', 'L3', 'L4', 'L5'],
        'MiniGrid-LavaCrossingS9N2': ['L1', 'L2', 'L3', 'L4', 'L5']
    }
    # 'point_maze': {
    #     'PointMaze-FourRoomsEnvDense-11x11': ['L1', 'L2', 'L3', 'L4', 'L5'],
    #     'PointMaze-ObstaclesEnvDense-11x11': ['L1', 'L2', 'L3', 'L4', 'L5']
    # }
    # 'parking': {
    #     'Parking-S-14-PC-': ['L1', 'L2', 'L3', 'L4', 'L5'],
    #     'Parking-S-14-PC-': ['L1', 'L2', 'L3', 'L4', 'L5']
    # }
    # 'panda': {
    #     'PandaMyReachDense': ['L1', 'L2', 'L3', 'L4', 'L5'],
    #     'PandaMyReachDense': ['L1', 'L2', 'L3', 'L4', 'L5']
    # }
}
# for minigrid: #TODO assert these instead i the beggingning of the code before beginning with the actual threading
recognizers = ['ExpertBasedGraml']
# recognizers = ['Graql']

# for point_maze:
# recognizers = ['ExpertBasedGraml']
# recognizers = ['Draco']

# for parking:
# recognizers = ['GCGraml']
# recognizers = ['GCDraco']

# for panda:
# recognizers = ['GCGraml']
# recognizers = ['GCDraco']

n = 5  # Number of times to execute each task

# Function to read results from the result file
def read_results(res_file_path):
    with open(res_file_path, 'rb') as f:
        results = dill.load(f)
    return results

# Every thread worker executes this function.
def run_experiment(domain, env, task, recognizer, i, generate_new=False):
    cmd = f"python odgr_executor.py --domain {domain} --recognizer {recognizer} --env_name {env} --task {task} --collect_stats"
    print(f"Starting execution: {cmd}")
    try:
        res_file_path = get_experiment_results_path(domain, env, task, recognizer)
        res_file_path_txt = res_file_path + '.txt'
        i_res_file_path_txt = res_file_path + f'_{i}.txt'
        res_file_path_pkl = res_file_path + '.pkl'
        i_res_file_path_pkl = res_file_path + f'_{i}.pkl'
        if generate_new or (not os.path.exists(i_res_file_path_txt) or not os.path.exists(i_res_file_path_pkl)):
            if os.path.exists(i_res_file_path_txt) or os.path.exists(i_res_file_path_pkl):
                i_res_file_path_txt = i_res_file_path_txt.replace(f'_{i}', f'_{i}_new')
                i_res_file_path_pkl = i_res_file_path_pkl.replace(f'_{i}', f'_{i}_new')
            # every thread in the current process starts a new process which executes the command. the current thread waits for the subprocess to finish.
            process = subprocess.Popen(cmd, shell=True)
            process.wait()
            # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"Execution failed: {cmd}")
                print(f"Error: {result.stderr}")
                return None
            else:
                print(f"Finished execution successfully: {cmd}")
            file_lock = threading.Lock()
            with file_lock:
                os.rename(res_file_path_pkl, i_res_file_path_pkl)
                os.rename(res_file_path_txt, i_res_file_path_txt)
        else:
            print(f"File {i_res_file_path_txt} already exists. Skipping execution of {cmd}")
        return ((domain, env, task, recognizer), read_results(i_res_file_path_pkl))
    except Exception as e:
        print(f"Exception occurred while running experiment: {e}")
        return None

# Collect results
results = {}

# create an executor that manages a pool of threads.
# Note that any failure in the threads will not stop the main thread from continuing and vice versa, nor will the debugger view the failure if in debug mode.
# Use prints and if any thread's printing stops suspect failure. If failure happened, use breakpoints before failure and use the watch to see the failure by pasting the problematic piece of code.
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for domain, envs in configs.items():
        for env, tasks in envs.items():
            for task in tasks:
                for recognizer in recognizers:
                    for i in range(n):
                        # submit returns a future object that represents the execution of the function by some thread in the pool. the method is added to the list and the executor will run it as soon as it can.
                        futures.append(executor.submit(run_experiment, domain, env, task, recognizer, i, generate_new=True if len(sys.argv) > 1 and sys.argv[1] == '--generate_new' else False))

    # main thread continues execution after the loop above. Here, it waits for all threads. every time a thread finishes, the main thread reads the results from the future object.
    # If debugging main thread, note the executor will stop creating new threads and running them since it exists and runs in the main thread.
    # probably, the main thread gets interrupted by the return of a future and knows to start execution of new threads and then continue main thread execution
    for future in concurrent.futures.as_completed(futures):
        # the objects returned by the 'result' func are tuples with key being the args inserted to 'submit'.
        if future.result() is None:
            print(f"for future {future}, future.result() is None. Continuing to next future.")
            continue
        key, result = future.result()
        print(f"main thread reading results from future {key}")
        # list because every experiment is executed n times.
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
        if percentage == 'total':
            continue
        if percentage not in compiled_accuracies[(domain, recognizer)].keys(): compiled_accuracies[(domain, recognizer)][percentage] = {}
        if percentage not in detailed_summary[key].keys(): detailed_summary[key][percentage] = {}
        consecutive_accuracies = [result[percentage]['consecutive']['accuracy'] for result in result_list] # accuracies in all different n executions
        non_consecutive_accuracies = [result[percentage]['non_consecutive']['accuracy'] for result in result_list]
        if 'consecutive' in compiled_accuracies[(domain, recognizer)][percentage].keys():
            compiled_accuracies[(domain, recognizer)][percentage]['consecutive'].extend(consecutive_accuracies)
        else:
            compiled_accuracies[(domain, recognizer)][percentage]['consecutive'] = consecutive_accuracies
        if 'non_consecutive' in compiled_accuracies[(domain, recognizer)][percentage].keys():
            compiled_accuracies[(domain, recognizer)][percentage]['non_consecutive'].extend(non_consecutive_accuracies)
        else:
            compiled_accuracies[(domain, recognizer)][percentage]['non_consecutive'] = non_consecutive_accuracies
        avg_consecutive_accuracy = np.mean(consecutive_accuracies)
        consecutive_std_dev = np.std(consecutive_accuracies)
        detailed_summary[key][percentage]['consecutive'] = (avg_consecutive_accuracy, consecutive_std_dev)
        avg_non_consecutive_accuracy = np.mean(non_consecutive_accuracies)
        non_consecutive_std_dev = np.std(non_consecutive_accuracies)
        detailed_summary[key][percentage]['non_consecutive'] = (avg_non_consecutive_accuracy, non_consecutive_std_dev)
     
compiled_summary = {}   
for key, percentage_dict in compiled_accuracies.items():
    compiled_summary[key] = {}
    for percentage, cons_accuracies in percentage_dict.items():
        compiled_summary[key][percentage] = {}
        for is_cons, accuracies in cons_accuracies.items():
            avg_accuracy = np.mean(accuracies)
            std_dev = np.std(accuracies)
            compiled_summary[key][percentage][is_cons] = (avg_accuracy, std_dev)

# Write different summary results to different files
detailed_summary_file_path = os.path.join('summaries',f"detailed_summary_{''.join(configs.keys())}_{recognizers[0]}.txt")
compiled_summary_file_path = os.path.join('summaries',f"compiled_summary_{''.join(configs.keys())}_{recognizers[0]}.txt")
with open(detailed_summary_file_path, 'w') as f:
    for key, percentage_dict in detailed_summary.items():
        domain, env, task, recognizer = key
        f.write(f"{domain}\t{env}\t{task}\t{recognizer}\n")
        for percentage, cons_info in percentage_dict.items():
            for is_cons, (avg_accuracy, std_dev) in cons_info.items():
                f.write(f"\t\t{percentage}\t{is_cons}\t{avg_accuracy:.4f}\t{std_dev:.4f}\n")
            
with open(compiled_summary_file_path, 'w') as f:
    for key, percentage_dict in compiled_summary.items():
        domain, recognizer = key
        f.write(f"{domain}\t{recognizer}\n")
        for percentage, cons_info in percentage_dict.items():
            for is_cons, (avg_accuracy, std_dev) in cons_info.items():
                f.write(f"\t\t{percentage}\t{is_cons}\t{avg_accuracy:.4f}\t{std_dev:.4f}\n")

print(f"Detailed summary results written to {detailed_summary_file_path}")
print(f"Compiled summary results written to {compiled_summary_file_path}")