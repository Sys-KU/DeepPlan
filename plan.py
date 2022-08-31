import torch
import sys
import util
import models
import pickle
import os.path
import numpy as np
import argparse
import logging
import copy
import time
from collections import OrderedDict
from typing import Tuple
from proto.deepplan_pb2 import ModelConfig, Plan, ModelInput, DataType

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser(description='DeepPlan Planner')
parser.add_argument('--model_name', '-m', type=str, required=True)
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--plan_dir', '-p', type=str, required=True)
parser.add_argument('--profile', action='store_true', required=False)
parser.add_argument('--trace', action='store_true', required=False)

args = parser.parse_args()

num_test = 100
num_warmup = 10

class MeasureRecorder():
    def __init__(self):
        self.events = OrderedDict()

    def record_start(self, key, stream=None):
        if key in self.events.keys():
            print(key)
            print("[Warning] The previous event tuple can be removed")
        event1 = torch.cuda.Event(enable_timing=True)
        event2 = torch.cuda.Event(enable_timing=True)

        event_tuple = (event1, event2)
        self.events[key] = (event_tuple)
        event1.record(stream)

    def record_end(self, key, stream=None):
        if not key in self.events.keys():
            raise RuntimeError("record_start() should be called before call record_end()")

        event1, event2 = self.events[key]
        event2.record(stream)

    def result(self):
        times = []
        for k, v in self.events.items():
            event1, event2 = v
            event1.synchronize()
            event2.synchronize()

            t = event1.elapsed_time(event2)
            times.append(t)

        return times

    def reset(self):
        self.events = {}

def print_layer_state_table(naive_layers, static_layers, dynamic_layers):
    boundary_lines = "=" * 78
    hyphens = "|{:<25}|{:<25}|{}".format('-'*25, '-'*25, '-'*25)
    print(boundary_lines)
    print("|{:<25}|{:<25}|{}".format("Layer", "Initial approach", "DeepPlan (DHA)"))
    print(hyphens)
    for i, layer in enumerate(naive_layers):
        layer_name = "{}-{}".format(layer['index'], layer['layer_type'])
        load_naive_layer = "O" if layer['exec_type'] == 0 else "X"
        load_static_layer = "O" if static_layers[i]['exec_type'] == 0 else "X"
        load_dynamic_layer = "O" if dynamic_layers[i]['exec_type'] == 0 else "X"

        size = layer['size']
        size /= (1024*1024)

        print("|{:<20} ({:.3f} MB) |{} {:<25}|{} {}".format(layer_name, size,
                                               load_static_layer,
                                               "(direct-host-access)" if load_naive_layer != load_static_layer else "",
                                               load_dynamic_layer,
                                               "(direct-host-access)" if load_naive_layer != load_dynamic_layer else ""
                                               )),
    print(boundary_lines)

def measure_load_layers(model):
    measure_rec = MeasureRecorder()

    load_times_list = []
    load_times = []

    layers = util.travel_layers(model)
    for step in range(num_warmup+num_test):
        for layer in layers:
            layer.cpu()
            layer.pin_memory()

        for layer in layers:
            measure_rec.record_start(layer.__qualname__)

            layer.to("cuda", non_blocking=True)
            measure_rec.record_end(layer.__qualname__)

        torch.cuda.synchronize()
        if step >= num_warmup:
            load_times_list.append(measure_rec.result())
        measure_rec.reset()

    load_times_list = np.array(load_times_list)
    load_times = load_times_list.sum(axis=0)
    load_times = load_times / num_test

    return load_times

def measure_exec_layers(model, x):
    measure_rec = MeasureRecorder()

    exec_times_list = []
    exec_times = []
    hooks = []

    def record_start(self, input):
        measure_rec.record_start(self.__qualname__)

    def record_end(self, input, output):
        measure_rec.record_end(self.__qualname__)

    layers = util.travel_layers(model)
    for layer in layers:
        if len(layer._forward_pre_hooks) == 0:
            pre_handle = layer.register_forward_pre_hook(record_start)
            handle = layer.register_forward_hook(record_end)
            hooks.append((pre_handle, handle))

    for step in range(num_warmup+num_test):
        with torch.no_grad():
            if type(x) is dict:
                model.forward(**x)
            else:
                model.forward(x)

        torch.cuda.synchronize()

        if step >= num_warmup:
            exec_times = measure_rec.result()
            exec_times_list.append(exec_times)
        measure_rec.reset()

    for hook in hooks:
        hook[0].remove()
        hook[1].remove()

    exec_times_list = np.array(exec_times_list)
    exec_times = exec_times_list.sum(axis=0)
    exec_times = exec_times/num_test

    return exec_times

def dump_profile_info(model, x, file_name):
    # Measure Load Time
    _layers = util.travel_layers(model)
    layer_load_times = measure_load_layers(model)

    # Measure GPU in-memory Exec time
    model.cuda()
    layer_cuda_exec_times = measure_exec_layers(model, x)

    # Measure GPU Direct Access Exec Time
    model.cpu()
    model.cuda_host()
    layer_cuda_host_exec_times = measure_exec_layers(model, x)

    # Measure GPU Direct Access Exec Time with benchmark

    layer_info_list = []

    for i, layer in enumerate(_layers):
        size = 0
        for key, param in layer._parameters.items():
            if param is not None:
                size += np.prod(np.array(param.size())) * 4
        for key, buf in layer._buffers.items():
            if buf is not None:
                size += np.prod(np.array(param.size())) * 4

        layer_info = {
                        'index': i,
                        'layer_type': layer.__class__.__name__,
                        'size': size,
                        'load_time': layer_load_times[i],
                        'cuda_exec_time': layer_cuda_exec_times[i],
                        'cuda_host_exec_time': layer_cuda_host_exec_times[i],
                        'exec_type': 0 # 0: Load Then Execution, 1: Remote Direct Access Execution 2: CPU
                     }

        layer_info_list.append(layer_info)

    with open(file_name, 'wb') as f:
        pickle.dump(layer_info_list, f)

    return layer_info_list

def load_profile_info(file_name):
    layer_info_list = []

    with open(file_name, 'rb') as f:
        layer_info_list = pickle.load(f)

    return layer_info_list

def update_PEF(layers):
    ready_time = 0
    run_time   = 0

    traces = [] # Ready, Run, Stall

    traces.append([0, 0, 0])

    for i, layer in enumerate(layers):
        stall_time = 0
        ready_time, run_time, _ = traces[-1]
        if layer['exec_type'] == 0:
            ready_time -= (layer['load_time'])
            if ready_time < 0:
                stall_time += (-ready_time)
                run_time   += (-ready_time)
                ready_time = 0

            ready_time += layer['cuda_exec_time']
            run_time   += layer['cuda_exec_time']
        elif layer['exec_type'] == 1:
            ready_time += layer['cuda_host_exec_time']
            run_time   += layer['cuda_host_exec_time']

        traces.append([ready_time, run_time, stall_time])

    return traces

def generate_naive_layers(layers):
    naive_layers = copy.deepcopy(layers)

    # Naive
    for layer in naive_layers:
        size = layer['size']
        layer['exec_type'] = 0 if size > 0 else 1

    return naive_layers


def generate_static_plan(layers):
    static_layers = generate_naive_layers(layers)

    for layer in static_layers:
        layer_type = layer['layer_type']

        if layer['exec_type'] == 1: continue

        if ("BatchNorm" in layer_type or
            "Embedding" in layer_type):
            #"LayerNorm" in layer_name or
            layer['exec_type'] = 1
        elif layer['cuda_host_exec_time'] < (layer['cuda_exec_time'] + layer['load_time']):
            layer['exec_type'] = 1

    return static_layers


def generate_dynamic_plan(layers):
    dynamic_layers = generate_naive_layers(layers)

    traces = update_PEF(dynamic_layers) 

    def sort_func(x):
        perf_gap = x['cuda_host_exec_time'] - x['cuda_exec_time']
        load_time = x['load_time']
        return (perf_gap, -load_time)
        #return -load_time/perf_gap

    # Generating layer execution plan
    for t in range(len(traces)):
        stall_time = traces[t][2] # Stall time in the corresponding layer.
        if stall_time <= 0: continue

        sorted_layers = sorted(dynamic_layers[:t], key = sort_func)

        for layer in sorted_layers[:t]:
            if layer['exec_type'] == 1: continue

            # Increased running time by convert the layer from load-then-execution to direct-host-access
            perf_gap = layer['cuda_host_exec_time']-layer['cuda_exec_time']

            is_overload = False
            if layer['cuda_host_exec_time'] > (layer['cuda_exec_time'] + 1.5*layer['load_time']):
                is_overload = True

            should_convert_DA = True
            # When the layer runs with direct-host-access, The load is heavy.
            if is_overload is True : should_convert_DA = False

            # The run time is longer than the current stall time.
            elif stall_time < perf_gap: should_convert_DA = False  

            if should_convert_DA is False: break

            index = layer['index']
            dynamic_layers[index]['exec_type'] = 1

            stall_time = stall_time - layer['load_time'] - perf_gap
            if stall_time <= 0:
                traces = update_PEF(dynamic_layers)
                break

    traces = update_PEF(dynamic_layers)

    return dynamic_layers

def generate_trace_module(model, x):
    layers = util.travel_layers(model)
    def hook(self, input: Tuple[torch.Tensor]):
        for name, param in self.named_parameters():
            if name in ['weight']:
                torch.sync_tensor_(param, input[0])

#        for parm in self.parameters():
#            torch.sync_tensor_(parm, input[0])
#        for buff in self.buffers():
#            torch.sync_tensor_(buff, input[0])

        return input

    for layer in layers:
        if len(layer._forward_pre_hooks) == 0:
            layer.register_forward_pre_hook(hook)

    trace_module = torch.jit.trace(model, x)
    return trace_module

def generate_plan(model, x, output_dir_path, do_profile=False, do_trace=False):
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)

    profile_info_path = os.path.join(
                            output_dir_path,
                            f'model_batch_{batch_size}.pickle'
                        )
    layers = []
    if (os.path.isfile(profile_info_path) is False) or (do_profile is True):
        logging.info("Dumping profile info")

        t1 = time.time()
        layers = dump_profile_info(model, x, profile_info_path)
        t2 = time.time()

        logging.info(f"Measurement time for profiling: {(t2-t1)*1e3:.3f} ms")
        logging.info("Dump completed")
    else:
        logging.info("Load profile info")
        layers = load_profile_info(profile_info_path)
        logging.info("Load completed")

    logging.info("Creating plans")
    naive_layers = generate_naive_layers(layers)
    static_layers = generate_static_plan(layers)

    t1 = time.time()
    dynamic_layers = generate_dynamic_plan(layers)
    t2 = time.time()
    logging.info("All plans are generated")

    print_layer_state_table(naive_layers, static_layers, dynamic_layers)
    logging.info(f"Plan Generating Time: {(t2-t1)*1e3:.3f} ms")

    model_config = ModelConfig()
    model_config.model_name = model_name

    def addInput(model_config, input_data):
        model_input = ModelInput()
        dtype = input_data.dtype
        if dtype == torch.float32:
            model_input.data_type = DataType.TYPE_FP32
        elif dtype == torch.int64:
            model_input.data_type = DataType.TYPE_INT64
        model_input.shape[:] = input_data.size()[1:]

        model_config.inputs.append(model_input)


    def addPlan(model_config, layers, plan_type):
        plan = Plan()
        plan.plan_type = plan_type;
        load_layers = []

        for layer in layers:
            if layer['exec_type'] == 0:
                load_layers.append(layer['index'])

        plan.load_layers[:] = load_layers
        model_config.plans.append(plan)

    addInput(model_config, input_data)
    for layers, plan_type in zip(
                                [static_layers, dynamic_layers, dynamic_layers],
                                [Plan.PlanType.STATIC,
                                 Plan.PlanType.DYNAMIC,
                                 Plan.PlanType.BENCH_DYNAMIC]):
        addPlan(model_config, layers, plan_type)


    util.write_to_pbtxt(model_config, os.path.join(output_dir_path, 'config.pbtxt'))
    logging.info("Model config is created")

    for d in range(torch.cuda.device_count()):
        trace_module_path = os.path.join(output_dir_path, f'model{d}.pt')
        if (os.path.isfile(trace_module_path) is False) or (do_trace is True):
            model.cuda(d)
            x = x.cuda(d)
            logging.info("Creating trace module")

            trace_module = generate_trace_module(model, x)

            logging.info("trace module is created")
            logging.info(f"Saving trace module to 'model{d}.pt'")
            torch.jit.save(trace_module, trace_module_path)
            logging.info("Saving completed")


if __name__ == "__main__":
    model_name = args.model_name
    batch_size = args.batch_size
    plan_dir   = args.plan_dir
    do_profile = args.profile
    do_trace   = args.trace

    plan_dir_path = os.path.join(os.getcwd(), plan_dir)
    output_dir_path = os.path.join(plan_dir_path, model_name)

    model = models.import_model(model_name)
    model.eval()

    input_data = models.import_data(model_name, batch_size)
    input_data = input_data.cuda()

    generate_plan(model, input_data, output_dir_path, do_profile, do_trace)
