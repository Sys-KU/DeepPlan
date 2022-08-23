# DeepPlan

An inference execution planner minimizes the performance penalty while provisioning DL models from host to GPU.

## 1.System overview of DeepPlan
![System overview of DeepPlan](figs/arch-eps-converted-to.png)

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

```
git clone --recursive https://github.com/pytorch/pytorch -b v1.9.0
cd pytorch
```

To simplify the step reflecting the changes on the framework, we have provided a patch file for DeepPlan.
The following command applies the patch to the PyTorch v1.9.0
```
patch -p1 < <DeepPlan_PATH>/pytorch.patch
```

After applying the patch file, let's compile the PyTorch.
```
python3 setup install
```

In addition to PyTorch, install pip modules using the command below, from DeepPlan's `Home` directory.
```
pip3 install -r requirements.txt
```

### 2.4 Build DeepPlan

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/pytorch ..
make
```

## 3. Step 1 ~ 2: Generating Layer Execution Plan

You need to create a plan from our Planner. In this tutorial, our target is ResNet50.
The `plan.py` Python module already imports the pre-trained model so that you can simply type the name of the model. 
```bash
# Create Plan
python3 plan.py -m resnet50 -p <plans_repository>
# The generated plan from this command is saved plans_repository
```

If you want to take a look at plans generated, you can click the following links.

* Plans

DeepPlan coordinates layer load and execution timing based on the corresponding plan.

## 4. Step3: Model Execution
If you created the plan above, you can run the model inference for the ResNet50 model with DeepPlan engine through the commands below, from DeepPlan's `Home` directory.
We provide three execution methods explained in our paper.

* DeepPlan (DHA)

```bash
./build/benchmark -m resnet50 -e deepplan
```
You should see output similar to the following:
```
Benchmarking Inference renset50
model average inference time : 11.2064 ms
```

* DeepPlan (DHA+PT)

```bash
./build/benchmark -m resnet50 -e deepplan -d 0 2 # d option represents the devices to be used for load.
```
You should see output similar to the following:
```
Benchmarking Inference renset101
model average inference time : 8.7267 ms
```

* On-Demand

```bash
./build/benchmark -m resnet50 -e demand
```
You should see output similar to the following:
```
Benchmarking Inference resnet50
model average inference time : 17.6628 ms
```

* PipeSwtich (Bai et al. OSDI 2020)

```bash
./build/benchmark -m resnet50 -e pipeline
```

You should see output similar to the following:
```
Benchmarking Inference resnet50
model average inference time : 12.2287 ms
```

## 5. Run Experiments

### 5.1 Figure 10: Performance comparison of DeepPlan and previous studies

### 5.2 Figure 12: 99% latency, goodput, and cold-start rate for BERT-Base (Synthetic workloads)

### 5.3 Figure 13: 99% latency for BERT-Large and GPT2 (Synthetic workloads)

### 5.4 Figure 14: Performance of real-world trace (Real-world workloads)
