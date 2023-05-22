import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import os
from tqdm import tqdm

from tree_sim import MEMORIES_PER_END_NODE, EXPIRATION_TIME

# This file contains the code that draws the graphs that you will find in example_graphs

# This function performs binned averaging for the dynamic_sim graphs
def normalize_by_time(time, value, round_to=1):
    out_time = []
    out_value = []
    count = []
    for i in range(len(value)):
        index = time[i] // round_to
        last_index = len(out_time) - 1
        if index > last_index:
            diff = index - last_index
            out_time.extend([(j + last_index) * round_to for j in range(diff)])
            out_value.extend([0] * diff)
            count.extend([1] * diff)
        else:
            count[index] += 1
        out_value[index] += value[i]

    out_time = np.array(out_time) + round_to * 2 # add 1*round_to to correct for an "off by one" in the above code, and another 1 to make datapoints be plotted at the end of their data range
    count = np.array(count)
    out_value = np.array(out_value) / count
    #print(f"Average data per bin was {np.mean(count)}")
    return out_time, out_value

class SmartTimer:
    def __init__(self):
        self.value = 0

    def tick(self, clock):
        tock = timer()
        self.value += tock - clock
        return tock




# These two functions are used by the plot functions to navigate the directories that store the raw data from the simulations.

def iterate_layer_dict(base_dir, var_num, var_str):
    for root, dirs, files in os.walk(f"{base_dir}/{var_str}_{var_num}"):
        for file in files:
            if file.endswith(".data"):
                with open(os.path.join(root,file), "rb") as f:
                    yield pickle.load(f)


def iterate_directory(base_dir):
    for root, dirs, files in os.walk(f"{base_dir}"):
        for file in files:
            if file.endswith(".data"):
                with open(os.path.join(root, file), "rb") as f:
                    yield pickle.load(f)


# Plot success rate vs request rate, with the option for either n or b to be displayed in the legend.
def plot_success_rate(base_dir, layer_num, detail):
    request_success_rate = []
    stderr_success_rate = []
    p = []
    data_arr = list(iterate_layer_dict(base_dir, layer_num, detail))
    if len(data_arr) == 0:
        raise ValueError
    data_arr.sort(key=lambda data: data["init_data"][0])
    for data in data_arr:
        if "samples" in data:
            sub_means = np.array([np.mean(np.array(subdata["request_success"])) for subdata in data["individual_samples"]])
        else:
            sub_means = np.array(data["request_success"])
        request_success_rate.append(np.mean(sub_means))
        stderr_success_rate.append(np.std(sub_means) / np.sqrt(np.shape(sub_means)[0]))

        p.append(data["init_data"][0])

    request_success_rate = np.array(request_success_rate)
    stderr_success_rate = np.array(stderr_success_rate) * 1.96 # +/- 1.96 * stderr gives a 95% confidence interval.  1.64 would be 90%.

    p = np.array(p)
    plt.fill_between(p, request_success_rate - stderr_success_rate, request_success_rate + stderr_success_rate, alpha=0.5)
    if detail == "n":
        plt.plot(p, request_success_rate, label=f"$N$={4 ** (layer_num-1)}")
    elif detail == "b":
        plt.plot(p, request_success_rate, label=f"$b$={layer_num}")

# Plot request time vs request rate, with the option for either n or b to be displayed in the legend.
def plot_request_time(base_dir, layer_num, whiskers, detail):
    if whiskers:
        request_times = []
        p = []
        data_arr = list(iterate_layer_dict(base_dir, layer_num, detail))
        if len(data_arr) == 0:
            raise ValueError
        data_arr.sort(key=lambda data: data["init_data"][0])
        for data in data_arr:
            cycles = data["request_cycles"]
            request_times.append(np.array(cycles))
            p.append(data["init_data"][0])

        #p = np.array(p)
        #plt.plot(p, mean_request_time, label=f"N={4 ** layer_num}")
        plt.boxplot(request_times)
    else:
        mean_request_time = []
        stderr_request_time = []
        p = []
        data_arr = list(iterate_layer_dict(base_dir, layer_num, detail))
        if len(data_arr) == 0:
            raise ValueError
        data_arr.sort(key=lambda data: data["init_data"][0])
        for data in data_arr:
            if "samples" in data:
                sub_means = np.array([np.mean(np.array(subdata["request_cycles"])) for subdata in data["individual_samples"]])
            else:
                sub_means = np.array(data["request_cycles"])
            mean_request_time.append(np.mean(sub_means))
            stderr_request_time.append(np.std(sub_means) / np.sqrt(np.shape(sub_means)[0]))
            p.append(data["init_data"][0])

        mean_request_time = np.array(mean_request_time)
        stderr_request_time = np.array(stderr_request_time) * 1.96 # +/- 1.96 * stderr gives a 95% confidence interval.  1.64 would be 90%.

        p = np.array(p)
        plt.fill_between(p, mean_request_time-stderr_request_time, mean_request_time+stderr_request_time, alpha=0.5)
        if detail == "n":
            plt.plot(p, mean_request_time, label=f"$N$={4 ** (layer_num-1)}")
        elif detail == "b":
            plt.plot(p, mean_request_time, label=f"$b$={layer_num}")

# Plot success rate or request time vs simulation time, with bin-averaged simulation results.  This is intended to be used with dynamic_sim.
def plot_time_sweep(base_dir, plotdict):
    result_time = []
    result_cycles = []
    result_rate = []
    result_buffer = []
    samples = 0

    # gather data from the files
    for data in iterate_directory(base_dir):
        samples += 1
        new_result_time = data["request_times"]
        new_result_cycles = data["request_cycles"]
        new_result_rate = data["request_success"]
        new_result_buffer = data["entanglement_buffer"]


        old_i = 0
        merged_result_time = []
        merged_result_cycles = []
        merged_result_rate = []

        for i in range(len(new_result_time)):
            curr_time = new_result_time[i]
            while old_i < len(result_rate) and result_time[old_i] <= curr_time:
                merged_result_time.append(result_time[old_i])
                merged_result_cycles.append(result_cycles[old_i])
                merged_result_rate.append(result_rate[old_i])
                old_i += 1

            merged_result_time.append(curr_time)
            merged_result_cycles.append(new_result_cycles[i])
            merged_result_rate.append(new_result_rate[i])

        merged_result_buffer = []
        for i in range(len(new_result_buffer)):
            for i2 in range(i * (samples - 1), (i+1) * (samples - 1)):
                merged_result_buffer.append(result_buffer[i2])
            merged_result_buffer.append(new_result_buffer[i])



        result_time = merged_result_time
        result_cycles = merged_result_cycles
        result_rate = merged_result_rate
        result_buffer = merged_result_buffer

    # read the input parameters
    normalize_bins = 64
    if "binsize" in plotdict:
        normalize_bins = plotdict["binsize"]

    yaxis = plotdict["y"]

    if yaxis in ["rate","r","s","success"]:
        yaxis = "rate"
    elif yaxis in ["cycles", "time", "t", "c"]:
        yaxis = "time"

    if yaxis == "rate":
        result_time, result_cycles = normalize_by_time(result_time, result_rate, normalize_bins)
        result_time = np.array(result_time, dtype='float64') / np.array([1000.0], dtype='float64')
        plt.plot(result_time, result_cycles)
    elif yaxis == "time":
        result_time, result_rate = normalize_by_time(result_time, result_cycles, normalize_bins)
        result_time = np.array(result_time, dtype='float64') / np.array([1000.0], dtype='float64')
        plt.plot(result_time, result_rate)
    elif yaxis == "buffer":
        k = data["init_data"][1]
        n = data["init_data"][2]
        max_buffer = (k ** n) * n * MEMORIES_PER_END_NODE
        result_time, result_buffer = normalize_by_time([i // samples for i in range(len(result_buffer))], result_buffer, normalize_bins)
        # add a "T-1" point at 0 so matplotlib plots it with 0 as the y minimum
        # this is not innacurate so I think its a reasonable approach
        result_time = np.insert(result_time, 0, min(min(result_time) - 1, 0))
        result_buffer = np.insert(result_buffer, 0, 0)
        #result_buffer /= max_buffer
        #print(f"Again after: {result_buffer[0]}")
        result_time = np.array(result_time, dtype='float64') / np.array([1000.0], dtype='float64')
        plt.plot(result_time, result_buffer)


# This function takes a dictionary that specifies the details of what plot is needed, figures out which calls to the above functions need to be made, and includes additional matplotlib boilerplate code.
def make_plot_from_dict(plotdict):
    fig = plt.figure(figsize=(4,3))
    yaxis = plotdict["y"]
    if yaxis in ["rate", "r", "s", "success"]:
        yaxis = "success rate"
    elif yaxis in ["time", "t"]:
        yaxis = "latency"
    elif yaxis in ["buffer", "ebits"]:
        yaxis = "memory buffer (e-bits)"

    xaxis = plotdict["x"]
    if xaxis in ["p", "probability"]:
        xaxis = "p"
    elif xaxis in ["t", "time"]:
        xaxis = "t"

    whiskers = False
    if "whiskers" in plotdict:
        whiskers = plotdict["whiskers"]
    elif "w" in plotdict:
        whiskers = plotdict["w"]
    if whiskers is None:
        whiskers = True

    b = 1
    if "b" in plotdict:
        b = int(plotdict["b"])

    detail = "n"
    if "detail" in plotdict:
        detail = plotdict["detail"]

    outfile = None
    if "outdir" in plotdict:
        outfile = plotdict["outdir"]

    indir = "summary_dicts"
    if "indir" in plotdict:
        indir = plotdict["indir"]

    legend = False
    if "legend" in plotdict:
        legend = plotdict["legend"]

    if xaxis == "p":
        xaxis = "$p$"
        for layer_num in range(2,17):
            try:
                if yaxis == "success rate":
                    plot_success_rate(indir, layer_num, detail)
                elif yaxis == "latency":
                    plot_request_time(indir, layer_num, whiskers, detail)
            except ValueError:
                continue
        if not whiskers:
            plt.xscale("log")

    elif xaxis == "t":
        xaxis = "time ($10^3$ cycles)"
        plot_time_sweep(indir, plotdict)

    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

    # for request time and success rate, we know the min and max values are 0 and MEMORIES_PER_END_NODE or 1 respectively
    # for memory buffer it doesn't get anywhere near its theoretical max and I already include the min of 0 by prepending a T-1=0 data point
    if yaxis in ["success rate", "latency"]:
        ymin = 0
        ymax = 1 if yaxis == "success rate" else EXPIRATION_TIME

        ymarg = (ymax - ymin) * plt.margins()[1]
        plt.ylim(ymin - ymarg, ymax + ymarg)

        
        
    if legend:
        plt.legend()
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, transparent=True, bbox_inches="tight")
        plt.clf()

