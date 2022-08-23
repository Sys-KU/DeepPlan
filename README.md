# DeepPlan

Title: Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access

## 1.Experimental Environment
### 1.1 Hardware
* AWS P3.8xlarge instance
* GPU: NVIDIA V100 (16GB) x 4ea
* Memory: 244GB DDR4 DRAM
* CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
* NVLink 2.0
* PCIe 3.0

For EuroSys '23 Artifact Evaluation Committee, we can provide the AWS instance we used if you don't have any machine that satisfies the requirements. Let us know through the HotCRP portal.

### 1.2 Sofware requirements
* Operating system: Ubuntu 18.04
* CUDA v11.3
* CuDNN v8.2.1
* ProtoBuf v3.11.4
* PyTorch v1.9
* Matplotlib (for generating graphs)

## 2. Build software components

### 2.1 Dependent packages
* build-essential
```bash
$ sudo apt update
$ sudo apt install build-essential
```

* CUDA Toolkit v11.3 & CuDNN v8.2.1

DeepPlan works based on the pytorch DL framework.
The CUDA Toolkit & CuDNN Library are required to build PyTorch.

To install the CUDA Toolkit, see this following link.

https://developer.nvidia.com/cuda-11.3.0-download-archive

To install the CuDNN Library, see this following link.

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

* ProtoBuf v3.11.4

DeepPlan use the ProtoBuf library to serialize or deserialize plans.
So The ProtoBuf is required to build DeepPlan.

To install the ProtoBuf, see this following link.

https://github.com/protocolbuffers/protobuf/blob/main/src/README.md

### 2.2 PyTorch
To use DeepPlan, it is required to modify PyTorch (v1.9) framework. Let's download the PyTorch v1.9.0 package first.

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

### 2.3 DeepPlan

```bash
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/pytorch ..
$ make
```

## 3. Setup execution plans

You need to create a plan for a given model. In this tutorial, our target is ResNet50.
The python module, `plan.py`,  already imports the pre-trained models evaluated in the paper so that you can simply type the name of the model.
```bash
# Create Plan
$ cd <DeepPlan_PATH>
$ mkdir -p plans
$ python3 plan.py -m resnet50 -p plans
# The generated plan from this command is saved the plans directory
```

If you want to take a look at generated plans (Table 3 in the paper), you can click the following links.

* [Plans](https://github.com/csl-ajou/DeepPlan/tree/main/plans/V100)


## 4. Run benchmarks
Once DeepPlan generate the execution plan for a given model, you can run the model inference with the DeepPlan engine through the commands below, from DeepPlan's `Home` directory.
Here, we have an example for ResNet50. In this section, we describe how to run four different execution methods,
Baseline (on-demand), PipeSwitch, DeepPlan (DHA), DeepPlan (PT), and DeepPlan (PT+DHA), explained in our paper.

Before running the model inference, you have to set `PLAN_REPO` environment variable which represents where plans are stored.

```bash
# The plan repository should be the same as the path specified in above creating a plan
export PLAN_REPO="/<DeepPlan_PATH>/plans"
```

 * Baseline (on-demand)

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

* DeepPlan (DHA)

```bash
$ ./build/benchmark -m resnet50 -e deepplan
```
You should see output similar to the following:
```bash
Benchmarking Inference renset50
model average inference time : 11.2064 ms
```

* DeepPlan (PT)

```bash
$ ./build/benchmark -m resnet50 -e pipeline -d 0 2 # d option represents the devices to be used for load
```

* DeepPlan (DHA+PT)

```bash
$ ./build/benchmark -m resnet50 -e deepplan -d 0 2 # d option represents the devices to be used for load
```
You should see output similar to the following:
```bash
Benchmarking Inference renset101
model average inference time : 8.7267 ms
```

## 5. Reproduce results in the paper
For experiments, we should have the model plans. To simplify creating model plans,
we provide `create_all_plans.sh` shell script that makes all model plans used in the experiments.

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
