# DeepPlan

An inference execution planner minimizes the performance penalty while provisioning DL models from host to GPU.

## 1.System overview of DeepPlan
As depicted above, DeepPlan takes a pre-trained model as an input and generates
an execution plan that is utilized in the inference server.
Steps 1 and 2 are integrated into a single Python module.
We provide a tutorial for ResNet50 to generate an execution plan
guided by DeepPlan in the [Generating Layer Execution Plan](#3-step-1--2-generating-layer-execution-plan) section.
Step 3 is to execute the plan while provisioning the model from host to GPU. In [Model Execution](#4-step3-model-execution),
we demonstrate how the generated plan is used for inference.

## 2.Experiment Environment
### 2.1 Hardware
* AWS P3.8xlarge instance
* GPU: NVIDIA V100 (16GB) x 4ea
* Memory: 244GB DDR4 DRAM
* CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
* NVLink 2.0
* PCIe 3.0

### 2.2 Sofware requirements
* Operating system: Ubuntu 18.04
* CUDA v11.3
* CuDNN v8.2.1
* ProtoBuf v3.11.4 (https://github.com/protocolbuffers/protobuf)

### 2.3 (Requirement) Building PyTorch for DeepPlan
To enable DeepPlan, the modified PyTorch (v1.9) framework is required. Let's download PyTorch v1.9.0 first.

```bash
$ git clone --recursive https://github.com/pytorch/pytorch -b v1.9.0
$ cd pytorch
```

To simplify the step reflecting the changes on the framework, we have provided a patch file for DeepPlan.
The following command applies the patch to the PyTorch v1.9.0
```bash
$ patch -p1 < <DeepPlan_PATH>/pytorch.patch
```

After applying the patch file, let's compile the PyTorch.

```bash
$ python3 setup install
```

In addition to PyTorch, install pip modules using the command below, from DeepPlan's `Home` directory.
```bash
$ pip3 install -r requirements.txt
```

### 2.4 Build DeepPlan

```bash
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/pytorch ..
$ make
```

## 3. Step 1 ~ 2: Generating Layer Execution Plan

You need to create a plan from our Planner. In this tutorial, our target is ResNet50.
The `plan.py` Python module already imports the pre-trained model so that you can simply type the name of the model. 
```bash
# Create Plan
$ cd <DeepPlan_PATH>
$ mkdir -p plans
$ python3 plan.py -m resnet50 -p plans
# The generated plan from this command is saved the plans directory
```

If you want to take a look at plans generated, you can click the following links.

* Plans

DeepPlan coordinates layer load and execution timing based on the corresponding plan.

## 4. Step3: Model Execution
If you created the plan above, you can run the model inference for the ResNet50 model with DeepPlan engine through the commands below, from DeepPlan's `Home` directory.
We provide four execution methods explained in our paper.

Before running the model inference, you have to set `PLAN_REPO` environment variable which represents where plans are stored.

```bash
# The plan repository should be the same as the path specified in above creating a plan
export PLAN_REPO="/<DeepPlan_PATH>/plans"
```

* DeepPlan (DHA)

```bash
$ ./build/benchmark -m resnet50 -e deepplan
```
You should see output similar to the following:
```bash
Benchmarking Inference renset50
model average inference time : 11.2064 ms
```

* DeepPlan (DHA+PT)

```bash
$ ./build/benchmark -m resnet50 -e deepplan -d 0 2 # d option represents the devices to be used for load.
```
You should see output similar to the following:
```bash
Benchmarking Inference renset101
model average inference time : 8.7267 ms
```

* On-Demand

```bash
$ ./build/benchmark -m resnet50 -e demand
```
You should see output similar to the following:
```bash
Benchmarking Inference resnet50
model average inference time : 17.6628 ms
```

* PipeSwtich (Bai et al. OSDI 2020)

```bash
$ ./build/benchmark -m resnet50 -e pipeline
```

You should see output similar to the following:
```bash
Benchmarking Inference resnet50
model average inference time : 12.2287 ms
```

## 5. Run Experiments
For experiments, we should have the model plans. To simplify creating model plans,
we provide `create_all_plans.sh` shell script that make all model plans used in the experiments.

```bash
$ cd DeepPlan/scripts
$ export PLAN_REPO="/<DeepPlan_PATH>/plans/V100"
$ source create_all_plans.sh # the plan repository is created in PLAN_REPO path.
```
For all shell scripts, we should setup `PLAN_REPO` variable which represents plans repository.
We provided experiments scripts for figure #10, #12, #13, and #14.
Run the script in the `DeepPlan/scripts/fig#/run.sh` directory and the result will be logged in
the same directory. If the Matplotlib library was installed in you machine,
the graph will be drawn in `fig#.pdf`.

### 5.1 Figure 10: Performance comparison of DeepPlan and previous studies
We evaluate the inference latency with a single batch for On-Demand, PipeSwitch, DeepPlan(DHA),
DeepPlan (PT), and DeepPlan (PT+DHA). The results are normalized to On-Demand (Baseline).

```bash
$ cd DeepPlan/scripts/fig10
$ source run.sh
```

### 5.2 Figure 12: 99% latency, goodput, and cold-start rate for BERT-Base (Synthetic workloads)
We perform this experiment on a four-GPU server in an AWS instance.
This experiment measures the 99% latency, goodput, and cold-start for BERT-Base
while increasing the number of model instances concurrently running on the GPUs.

```bash
$ cd DeepPlan/scripts/fig12
$ source run.sh
```

### 5.3 Figure 13: 99% latency for BERT-Large and GPT2 (Synthetic workloads)
This experiment is similar to above the experiment (Figure 12) except that
the evaluation model is changed from BERt-base to Bert-Large and GPT2.
```bash
$ cd DeepPlan/scripts/fig13
$ source run.sh
```

### 5.4 Figure 14: Performance of real-world trace (Real-world workloads)
This experiment is also performed on a four-GPU server in an AWS instance.
The above experiments (Figure 12, Figure 13) run with synthetic trace. But
this experiment run with real-world trace derived from Microsoft Azure Funtions.
In this experiment, we evaluate three workloads of three hours each (total 9 hours).

To run this experiment, you should prepare auzre trace dataset.
https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md

The following command download the azure-trace dataset.
```bash
$ cd DeepPlan/scripts
$ source download_azure_trace_dataset.sh

# To recognize this trace file from client, The `AZURE_TRACE_DIR` variable should be set
$ export AZURE_TRACE_DIR="/<DeepPlan_PATH>/scripts/azure-functions"
```

```bash
$ cd DeepPlan/scripts/fig14
$ source run.sh
```

