# DeepPlan

An inference execution planner minimizes the performance penalty while provisioning DL models from host to GPU.

## 1.System overview of DeepPlan
![System overview of DeepPlan](figs/arch-eps-converted-to.png)

As depicted above, DeepPlan takes a pre-trained model as an input and generates
an execution plan that is utilized in the inference server.
Steps 1 and 2 are integrated into a single Python module.
We provide a tutorial for ResNet101 to generate an execution plan
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
The following command applies the patch to the stock PyTorch v1.9.0
```
patch -p1 < <DeepPlan_PATH>/pathfile
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
Then, you should see output similar to the following:

|Layer                    |Load? (Static)           |Load? (Adaptive)
|-------------------------|-------------------------|-------------------------
|0-Conv2d             (0.036 MB) |O                          |O
|1-BatchNorm2d        (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|2-ReLU               (0.000 MB) |X                          |X
|3-MaxPool2d          (0.000 MB) |X                          |X
|4-Conv2d             (0.016 MB) |O                          |O
|5-BatchNorm2d        (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|6-Conv2d             (0.141 MB) |X (direct-host-access)     |X (direct-host-access)
|7-BatchNorm2d        (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|8-Conv2d             (0.062 MB) |O                          |O
|9-BatchNorm2d        (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|10-ReLU              (0.000 MB) |X                          |X
|11-Conv2d            (0.062 MB) |O                          |O
|12-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|13-Conv2d            (0.062 MB) |O                          |O
|14-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|15-Conv2d            (0.141 MB) |X (direct-host-access)     |X (direct-host-access)
|16-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|17-Conv2d            (0.062 MB) |O                          |O
|18-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|19-ReLU              (0.000 MB) |X                          |X
|20-Conv2d            (0.062 MB) |O                          |O
|21-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|22-Conv2d            (0.141 MB) |X (direct-host-access)     |X (direct-host-access)
|23-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|24-Conv2d            (0.062 MB) |O                          |O
|25-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|26-ReLU              (0.000 MB) |X                          |X
|27-Conv2d            (0.125 MB) |O                          |O
|28-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|29-Conv2d            (0.562 MB) |O                          |O
|30-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|31-Conv2d            (0.250 MB) |O                          |O
|32-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|33-ReLU              (0.000 MB) |X                          |X
|34-Conv2d            (0.500 MB) |O                          |O
|35-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|36-Conv2d            (0.250 MB) |O                          |O
|37-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|38-Conv2d            (0.562 MB) |X (direct-host-access)     |X (direct-host-access)
|39-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|40-Conv2d            (0.250 MB) |O                          |O
|41-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|42-ReLU              (0.000 MB) |X                          |X
|43-Conv2d            (0.250 MB) |O                          |O
|44-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|45-Conv2d            (0.562 MB) |X (direct-host-access)     |X (direct-host-access)
|46-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|47-Conv2d            (0.250 MB) |O                          |O
|48-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|49-ReLU              (0.000 MB) |X                          |X
|50-Conv2d            (0.250 MB) |O                          |O
|51-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|52-Conv2d            (0.562 MB) |X (direct-host-access)     |X (direct-host-access)
|53-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|54-Conv2d            (0.250 MB) |O                          |O
|55-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|56-ReLU              (0.000 MB) |X                          |X
|57-Conv2d            (0.500 MB) |O                          |O
|58-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|59-Conv2d            (2.250 MB) |O                          |O
|60-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|61-Conv2d            (1.000 MB) |O                          |O
|62-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|63-ReLU              (0.000 MB) |X                          |X
|64-Conv2d            (2.000 MB) |O                          |O
|65-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|66-Conv2d            (1.000 MB) |O                          |O
|67-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|68-Conv2d            (2.250 MB) |O                          |O
|69-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|70-Conv2d            (1.000 MB) |O                          |O
|71-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|72-ReLU              (0.000 MB) |X                          |X
|73-Conv2d            (1.000 MB) |O                          |O
|74-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|75-Conv2d            (2.250 MB) |X (direct-host-access)     |O
|76-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|77-Conv2d            (1.000 MB) |O                          |O
|78-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|79-ReLU              (0.000 MB) |X                          |X
|80-Conv2d            (1.000 MB) |O                          |O
|81-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|82-Conv2d            (2.250 MB) |X (direct-host-access)     |O
|83-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|84-Conv2d            (1.000 MB) |O                          |O
|85-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|86-ReLU              (0.000 MB) |X                          |X
|87-Conv2d            (1.000 MB) |O                          |O
|88-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|89-Conv2d            (2.250 MB) |X (direct-host-access)     |O
|90-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|91-Conv2d            (1.000 MB) |O                          |O
|92-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|93-ReLU              (0.000 MB) |X                          |X
|94-Conv2d            (1.000 MB) |O                          |O
|95-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|96-Conv2d            (2.250 MB) |X (direct-host-access)     |O
|97-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|98-Conv2d            (1.000 MB) |O                          |O
|99-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|100-ReLU             (0.000 MB) |X                          |X
|101-Conv2d           (1.000 MB) |O                          |O
|102-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|103-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|104-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|105-Conv2d           (1.000 MB) |O                          |O
|106-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|107-ReLU             (0.000 MB) |X                          |X
|108-Conv2d           (1.000 MB) |O                          |O
|109-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|110-Conv2d           (2.250 MB) |O                          |O
|111-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|112-Conv2d           (1.000 MB) |O                          |O
|113-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|114-ReLU             (0.000 MB) |X                          |X
|115-Conv2d           (1.000 MB) |O                          |O
|116-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|117-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|118-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|119-Conv2d           (1.000 MB) |O                          |O
|120-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|121-ReLU             (0.000 MB) |X                          |X
|122-Conv2d           (1.000 MB) |O                          |O
|123-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|124-Conv2d           (2.250 MB) |O                          |O
|125-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|126-Conv2d           (1.000 MB) |O                          |O
|127-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|128-ReLU             (0.000 MB) |X                          |X
|129-Conv2d           (1.000 MB) |O                          |O
|130-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|131-Conv2d           (2.250 MB) |O                          |O
|132-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|133-Conv2d           (1.000 MB) |O                          |O
|134-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|135-ReLU             (0.000 MB) |X                          |X
|136-Conv2d           (1.000 MB) |O                          |O
|137-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|138-Conv2d           (2.250 MB) |O                          |O
|139-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|140-Conv2d           (1.000 MB) |O                          |O
|141-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|142-ReLU             (0.000 MB) |X                          |X
|143-Conv2d           (1.000 MB) |O                          |O
|144-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|145-Conv2d           (2.250 MB) |O                          |O
|146-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|147-Conv2d           (1.000 MB) |O                          |O
|148-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|149-ReLU             (0.000 MB) |X                          |X
|150-Conv2d           (1.000 MB) |O                          |O
|151-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|152-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|153-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|154-Conv2d           (1.000 MB) |O                          |O
|155-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|156-ReLU             (0.000 MB) |X                          |X
|157-Conv2d           (1.000 MB) |O                          |O
|158-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|159-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|160-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|161-Conv2d           (1.000 MB) |O                          |O
|162-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|163-ReLU             (0.000 MB) |X                          |X
|164-Conv2d           (1.000 MB) |O                          |O
|165-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|166-Conv2d           (2.250 MB) |O                          |O
|167-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|168-Conv2d           (1.000 MB) |O                          |O
|169-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|170-ReLU             (0.000 MB) |X                          |X
|171-Conv2d           (1.000 MB) |O                          |O
|172-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|173-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|174-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|175-Conv2d           (1.000 MB) |O                          |O
|176-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|177-ReLU             (0.000 MB) |X                          |X
|178-Conv2d           (1.000 MB) |O                          |O
|179-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|180-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|181-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|182-Conv2d           (1.000 MB) |O                          |O
|183-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|184-ReLU             (0.000 MB) |X                          |X
|185-Conv2d           (1.000 MB) |O                          |O
|186-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|187-Conv2d           (2.250 MB) |O                          |O
|188-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|189-Conv2d           (1.000 MB) |O                          |O
|190-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|191-ReLU             (0.000 MB) |X                          |X
|192-Conv2d           (1.000 MB) |O                          |O
|193-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|194-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|195-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|196-Conv2d           (1.000 MB) |O                          |O
|197-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|198-ReLU             (0.000 MB) |X                          |X
|199-Conv2d           (1.000 MB) |O                          |O
|200-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|201-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|202-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|203-Conv2d           (1.000 MB) |O                          |O
|204-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|205-ReLU             (0.000 MB) |X                          |X
|206-Conv2d           (1.000 MB) |O                          |O
|207-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|208-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|209-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|210-Conv2d           (1.000 MB) |O                          |O
|211-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|212-ReLU             (0.000 MB) |X                          |X
|213-Conv2d           (1.000 MB) |O                          |O
|214-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|215-Conv2d           (2.250 MB) |X (direct-host-access)     |O
|216-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|217-Conv2d           (1.000 MB) |O                          |O
|218-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|219-ReLU             (0.000 MB) |X                          |X
|220-Conv2d           (2.000 MB) |O                          |O
|221-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|222-Conv2d           (9.000 MB) |O                          |O
|223-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|224-Conv2d           (4.000 MB) |O                          |O
|225-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |X (direct-host-access)
|226-ReLU             (0.000 MB) |X                          |X
|227-Conv2d           (8.000 MB) |O                          |O
|228-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |X (direct-host-access)
|229-Conv2d           (4.000 MB) |O                          |O
|230-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|231-Conv2d           (9.000 MB) |O                          |O
|232-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|233-Conv2d           (4.000 MB) |O                          |O
|234-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|235-ReLU             (0.000 MB) |X                          |X
|236-Conv2d           (4.000 MB) |O                          |O
|237-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|238-Conv2d           (9.000 MB) |O                          |O
|239-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|240-Conv2d           (4.000 MB) |O                          |O
|241-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|242-ReLU             (0.000 MB) |X                          |X
|243-AdaptiveAvgPool2d (0.000 MB) |X                          |X
|244-Linear           (7.816 MB) |X (direct-host-access)     |O

DeepPlan coordinates layer load and execution timing based on the corresponding plan.

## 4. Step3: Model Execution
If you created the plan above, you can run the model inference for the ResNet101 model with DeepPlan engine through the commands below, from DeepPlan's `Home` directory. We provide three execution methods explained in our paper.

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
