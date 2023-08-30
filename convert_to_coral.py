import sys
import torch
import torchvision
import numpy as np
from pytorch2keras.converter import pytorch_to_keras
import tensorflow as tf
import pytorch_model_summary as pms
import tensorflow.lite as tflite
import network

#https://github.com/joaopauloschuler/neural-api/issues/67
class HardSigmoidTorch(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return torch.nn.functional.relu(torch.tensor(6.0))*(x + torch.tensor(3.0)) * (torch.tensor(1.0) / torch.tensor(6.0))
    
class HardSwishTorch(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigmoid = HardSigmoidTorch()
    
    def forward(self, x):
        return torch.mul(self.sigmoid(x), x)
    
class RecycleNetwork(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v2(weights='DEFAULT')
        #self.backbone.classifier[3] = torch.nn.Linear(1024, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

input_test = torch.rand(1, 3, 150, 150)
model = RecycleNetwork(5)
model.eval()
print(model)
def replace_hardswish(model):
    for layer in model.named_children():
        if isinstance(layer[1], torch.nn.Hardswish):
            setattr(model, layer[0], HardSwishTorch())
        elif isinstance(layer[1], torch.nn.Hardsigmoid):
            setattr(model, layer[0], HardSigmoidTorch())
        else:
            replace_hardswish(layer[1])
gallery_data = np.load('train_data.npz')
gallery_image = gallery_data['arr_0']
def representative_data_gen():
    for image in gallery_image:
        image = np.expand_dims(image,0)
        yield [image]


replace_hardswish(model)
# print(model)
# print(model(torch.rand(1,3,150,150)))
model = pytorch_to_keras(model, input_test, [(3, 150,150)], verbose=False, change_ordering=True,  name_policy='renumerate')
model.summary()
# model.input.set_shape((1,) + model.input.shape[1:])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_converter=True
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], torch.Tensor(1, 150, 150,3))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)