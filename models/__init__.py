import torch
from transformers import AutoModel
from enum import Enum
from typing import Union, List

class ModelType(Enum):
    CNN = 0
    TRANSFORMER = 1

class ModelConfig:
    def __init__(
        self,
        model_name: str,
        model_type: Union[str],
        input_shape: List[int],
        generate_input_func,
        max_num: int = -1,
        num_layers: int = 0,
    ):
        self.model_name = model_name

        if model_type in ("CNN", ModelType.CNN):
            self.model_type = ModelType.CNN
        elif model_type in ("Transformer", ModelType.TRANSFORMER):
            self.model_type = ModelType.TRANSFORMER

        self.input_shape = input_shape
        self.generate_input_func = generate_input_func
        self.max_num = max_num
        self.num_layers = num_layers


    def generateInput(self, batch_size):
        t = self.generate_input_func(batch_size, self.input_shape, self.max_num)
        return t

    def generateModel(self):
        model = None
        if self.model_type  == ModelType.CNN:
            model = torch.hub.load("pytorch/vision:v0.10.0", self.model_name)
        elif self.model_type == ModelType.TRANSFORMER:
            model = AutoModel.from_pretrained(self.model_name, torchscript=True)
            model.num_layers = self.num_layers
        model.is_parallel = False

        return model


def generate_CNN_input(batch_size, shape, max_num=0):
    input_shape = [batch_size] + shape
    x = torch.randn(input_shape)
    return x

def generate_TRS_input(batch_size, shape, max_num):
    input_shape = [batch_size] + shape
    x = torch.randint(max_num, input_shape, dtype=torch.long)
    return x

def generate_T5_input(batch_size, shape, max_num):
    x = generate_TRS_input(batch_size, shape, max_num)
    return {"input_ids": x, "decoder_input_ids": x}


model_list = {
        'vgg16': ModelConfig('vgg16', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'resnet50': ModelConfig('resnet50', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'resnet101': ModelConfig('resnet101', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'resnext50': ModelConfig('resnext50_32x4d', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'resnext101': ModelConfig('resnext101_32x8d', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'wide_resnet50': ModelConfig('wide_resnet50_2', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'wide_resnet101': ModelConfig('wide_resnet101_2', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'inception_v3': ModelConfig('inception_v3', ModelType.CNN, [3, 224, 224], generate_CNN_input),
        'bert_base': ModelConfig('bert-base-uncased', ModelType.TRANSFORMER, [384], generate_TRS_input, max_num=30522, num_layers=12),
        'bert_large': ModelConfig('bert-large-uncased', ModelType.TRANSFORMER, [384], generate_TRS_input, max_num=30522, num_layers=24),
        'gpt2': ModelConfig('gpt2', ModelType.TRANSFORMER, [1024], generate_TRS_input, max_num=50257, num_layers=12),
        'gpt2_medium': ModelConfig('gpt2-medium', ModelType.TRANSFORMER, [1024], generate_TRS_input, max_num=50257, num_layers=24),
        'bart_base': ModelConfig('facebook/bart-base', ModelType.TRANSFORMER, [384], generate_TRS_input, max_num=50265),
        't5_base': ModelConfig('t5-base', ModelType.TRANSFORMER, [300], generate_T5_input, max_num=32129, num_layers=12),
        't5_small': ModelConfig('t5-small', ModelType.TRANSFORMER, [300], generate_T5_input, max_num=32129, num_layers=6),
        't5_large': ModelConfig('t5-large', ModelType.TRANSFORMER, [300], generate_T5_input, max_num=32129, num_layers=24),
        'roberta_base': ModelConfig('roberta-base', ModelType.TRANSFORMER, [384], generate_TRS_input, max_num=50265, num_layers=12),
        'roberta_large': ModelConfig('roberta-large', ModelType.TRANSFORMER, [384], generate_TRS_input, max_num=50265, num_layers=24),
        }

def import_model(model_name):
    model = None

    if model_name in model_list:
        model = model_list[model_name].generateModel()
    else:
        raise RuntimeError(f"[Error] Not found '{model_name}' model")

    return model

def import_data(model_name, batch_size):
    data = None

    if model_name in model_list:
        data = model_list[model_name].generateInput(batch_size)
    else:
        raise RuntimeError(f"[Error] Not found '{model_name}' model")

    return data
