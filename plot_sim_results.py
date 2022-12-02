import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from tqdm import tqdm

# TODO: this file should really get a full rework

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

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

    out_time = np.array(out_time)
    count = np.array(count)
    out_value = np.array(out_value) / count
    print(f"Average data per bin was {np.mean(count)}")
    return out_time, out_value

class SmartTimer:
    def __init__(self):
        self.value = 0

    def tick(self, clock):
        tock = timer()
        self.value += tock - clock
        return tock

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

# full documentation of the file format
# they are pickled dictionaries of numpy data
# they are organized like this:
# outermost_dict = {
# "init_data" : (p, k, num_layers, num_cycles), # parameters for the simulated scenario
# "stats" : (requests_enqueued, requests_satisfied, requests_expired, entanglements_expired, entanglements_used, entanglements_made), # stats of how many requests/entanglements were expired vs used
# "request_cycles" : [array containing the number of cycles it took for each completed request to complete], # more detailed data about request timing, mainly used for whisker plots
# "timings" : (timer_memory_update, timer_request_make, timer_request_solve), # I've broken the main simulation loop into three segments and I record the timings for profiling purposes to help me optimize the code. I've already made the code ~2x faster with the help of this data.
# }


def plot_success_rate(base_dir,layer_num, detail):
    request_success_rate = []
    p = []
    data_arr = list(iterate_layer_dict(base_dir, layer_num, detail))
    if len(data_arr) == 0:
        raise ValueError
    data_arr.sort(key=lambda data: data["init_data"][0])
    for data in data_arr:
        stats = data["stats"]
        request_success_rate.append(stats[1]/stats[0])
        p.append(data["init_data"][0])

    request_success_rate = np.array(request_success_rate)
    p = np.array(p)
    if detail == "n":
        plt.plot(p, request_success_rate, label=f"N={4 ** layer_num}")
    elif detail == "b":
        plt.plot(p, request_success_rate, label=f"b={layer_num}")


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
        p = []
        data_arr = list(iterate_layer_dict(base_dir, layer_num, detail))
        if len(data_arr) == 0:
            raise ValueError
        data_arr.sort(key=lambda data: data["init_data"][0])
        for data in data_arr:
            cycles = data["request_cycles"]
            mean_request_time.append(np.mean(np.array(cycles)))
            p.append(data["init_data"][0])

        mean_request_time = np.array(mean_request_time)
        p = np.array(p)
        if detail == "n":
            plt.plot(p, mean_request_time, label=f"N={4 ** layer_num}")
        elif detail == "b":
            plt.plot(p, mean_request_time, label=f"b={layer_num}")


def plot_time_sweep(base_dir, plotdict):
    result_time = []
    result_cycles = []
    result_rate = []

    # gather data from the files
    for data in iterate_directory(base_dir):
        new_result_time = data["request_times"]
        new_result_cycles = data["request_cycles"]
        new_result_rate = data["request_success"]

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

        result_time = merged_result_time
        result_cycles = merged_result_cycles
        result_rate = merged_result_rate

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
        print(f"Length before: {len(result_cycles)}")
        result_time, result_cycles = normalize_by_time(result_time, result_rate, normalize_bins)
        plt.plot(result_time, result_cycles)
        print(f"printed length: {len(result_cycles)}")
    elif yaxis == "time":
        result_time, result_rate = normalize_by_time(result_time, result_cycles, normalize_bins)
        plt.plot(result_time, result_rate)

def make_plot_from_dict(plotdict):
    yaxis = plotdict["y"]
    if yaxis in ["rate", "r", "s", "success"]:
        yaxis = "Success Rate"
    elif yaxis in ["time", "t"]:
        yaxis = "Mean Request Time"

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
        if b > 1:
            xaxis = f"p / {b}"
        for layer_num in range(2,17):
            try:
                if yaxis == "Success Rate":
                    plot_success_rate(indir, layer_num, detail)
                elif yaxis == "Mean Request Time":
                    plot_request_time(indir, layer_num, whiskers, detail)
            except ValueError:
                continue
        if not whiskers:
            plt.xscale("log")

    elif xaxis == "t":
        xaxis = "Time (cycles)"
        plot_time_sweep(indir, plotdict)

    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if legend:
        plt.legend()
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, transparent=True)
        plt.clf()

