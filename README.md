# Ignite

Title: An Efficient DNN Model Serving System using Layer-wise Caching and Direct-Host-Access

Ignite is an extend project of our previous EuroSys'23 paper, "Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access".
This project includes a new scheme called Lcache, which efficiently manages GPU memory for caching DL models.

## 1.Experimental Environment
### 1.1 Hardware
* AWS P3.8xlarge instance
* GPU: NVIDIA V100 (16GB) x 4ea
* Memory: 244GB DDR4 DRAM
* CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
* NVLink 2.0
* PCIe 3.0

### 1.2 Software requirements
* Operating system: Ubuntu 22.04
* CUDA v12.1
* CuDNN v8.9.7
* ProtoBuf v3.13.0
* Boost v1.74
* TBB (Threading Building-Blocks) [v2021.5.0](https://github.com/uxlfoundation/oneTBB/tree/v2021.5.0)
* PyTorch v2.1
* Matplotlib v3.3.4 (for generating graphs)

## 2. Build software components

### 2.1 Dependent packages
* build-essential
```bash
$ sudo apt update
$ sudo apt install build-essential
```

* C++ Library on Ubuntu
```
$ sudo apt-get install libtbb-dev libboost-all-dev
```

* CUDA Toolkit v12.1 & CuDNN v8.9.7

Ignite works with the PyTorch DL framework. To run PyTorch,
we are supposed to install the dependent packages, CUDA and CuDNN.

To install the CUDA Toolkit, see this link: [Download Installer for Linux Ubuntu 22.04 x86_64](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)

To install the CuDNN Library, see this link: [Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html) and [CuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

* ProtoBuf v3.13.0

Ignite uses the ProtoBuf library to serialize or deserialize plans.
So, ProtoBuf is required to build Ignite. To install ProtoBuf, see this
following link: https://github.com/protocolbuffers/protobuf/blob/main/src/README.md

### 2.2 PyTorch
To use Ignite, it is required to modify PyTorch (v2.1) framework.
To simplify the step reflecting the code changes on the framework, we have provided a patch file for Ignite.
The following command applies the patch to the PyTorch v2.1.0.

```bash
$ cd $WORKSPACE
$ # Let's first clone the Ignite repository and set the path
$ git clone https://github.com/csl-ajou/DeepPlan -b ignite Ignite
$ IGNITE_HOME=$WORKSPACE/Ignite
$
$ # Let's download the PyTorch v2.1.0 package and set the path
$ git clone --recursive https://github.com/pytorch/pytorch -b v2.1.0
$ PYTORCH_HOME=$WORKSPACE/pytorch
$
$ cd $PYTORCH_HOME
$ patch -p1 < $IGNITE_HOME/pytorch.patch
```

After applying the patch file, let's compile the PyTorch.

```bash
$ python3 setup.py install
```

In addition to PyTorch, install pip modules using the command below, from Ignite's `Home` directory.
```bash
$ cd $IGNITE_HOME
$ pip3 install -r requirements.txt
```

### 2.3 Ignite

After successfully patching and building the PyTorch framework, we are
ready to build Ignite to generate inference execution plans and
the DL server prototype.

```bash
$ cd $IGNITE_HOME
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=$PYTORCH_HOME ..
$ make
```

## 3. Setup execution plans

You need to create a plan for a given model. In this tutorial, our target is BERT-Base.
The python module, `plan.py`,  already imports the pre-trained models evaluated in the paper so that you can simply type the name of the model.
```bash
# Create Plan
$ cd $IGNITE_HOME
$ mkdir -p plan_repo
$ python3 plan.py -m bert_base -p plan_repo
# The generated plan from this command is saved the plans directory
```

If you want to take a look at generated plans (Table 3 in the paper), you can click the following links.

* [Plans](https://github.com/csl-ajou/DeepPlan/tree/ignite/plans/V100)


## 4. Run benchmarks
Once Ignite generate the execution plan for a given model, you can run the model inference with the Ignite engine through the commands below, from Ignite's `Home` directory.
Here, we have an example for BERT-Base. In this section, we describe how to run three different execution methods,
Baseline (on-demand), PipeSwitch, and Ignite (DHA), explained in our paper.

Before running the model inference, you have to set `PLAN_REPO` environment variable which represents where plans are stored.

```bash
# The plan repository should be the same as the path specified in above creating a plan
$ export PLAN_REPO=$IGNITE_HOME/plan_repo
$ cd $IGNITE_HOME
```

 * Baseline (on-demand)

```bash
$ ./build/benchmark -m bert_base -e demand
```
You should see output similar to the following:
```bash
Benchmarking Inference bert_base
Average Latency : 48.7033 ms
```

* PipeSwtich (Bai et al. OSDI 2020)

```bash
$ ./build/benchmark -m bert_base -e pipeline
```

You should see output similar to the following:
```bash
Benchmarking Inference bert_base
Average Latency : 40.4792 ms
```

* DeepPlan (DHA)

```bash
$ ./build/benchmark -m bert_base -e deepplan
```
You should see output similar to the following:
```bash
Benchmarking Inference bert_base
Average Latency : 32.1664 ms
```

## 5. Reproduce results in the paper
To reproduce the experimental results presented in the paper, we should have the model plans. To simplify creating model plans,
we provide `create_all_plans.sh` shell script that makes all model plans used in the experiments.

```bash
$ cd $IGNITE_HOME/scripts
$ mkdir -p $IGNITE_HOME/plan_repo/V100
$ export PLAN_REPO=$IGNITE_HOME/plan_repo/V100
$ source create_all_plans.sh # the plan repository is created in PLAN_REPO path.
```
For all shell scripts, we should setup `PLAN_REPO` variable which represents plans repository.
We provided experiments scripts for figure #5 and #16.
Run the script in the `$IGNITE_HOME/scripts/fig#/run.sh` directory and the result will be logged in
the same directory. If the Matplotlib library was installed in your machine,
the graph will be drawn in `fig#.pdf`.

### 5.1 Figure 5: Performance analysis for Layer-wise Caching
This experiment shows cold-start latencies as the number of cached layers.
The result helps us to understand the performance impact of layer-wise caching.

```bash
$ cd $IGNITE_HOME/scripts/fig5
$ source run.sh
```

### 5.2 Figure 16: Performance of real-world trace (Real-world workloads)
This experiment is performed on a four-GPU server in an AWS instance
and runs with a real-world trace derived from Microsoft Azure Functions.
In this experiment, we evaluate three workloads of three hours each (total 9 hours).

To run this experiment, you should prepare azure trace dataset.
https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md

The following command download the azure-trace dataset.
```bash
$ cd $IGNITE_HOME/scripts
$ source download_azure_trace_dataset.sh

# To recognize this trace file from client, The `AZURE_TRACE_DIR` variable should be set
$ export AZURE_TRACE_DIR=$IGNITE_HOME/scripts/azure-functions
```

```bash
$ cd $IGNITE_HOME/scripts/fig14
$ source run.sh
```
