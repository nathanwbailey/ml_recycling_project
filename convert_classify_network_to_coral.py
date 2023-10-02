import sys
import torch
import torchvision
import numpy as np
from pytorch2keras.converter import pytorch_to_keras
import tensorflow as tf
import pytorch_model_summary as pms
import tensorflow.lite as tflite
from network import RecycleNetwork

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


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
    


input_test = torch.rand(1, 3, 150, 150)
model = RecycleNetwork(5)
model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.eval()

model = pytorch_to_keras(model, input_test, [(3, 150,150)], verbose=False, change_ordering=True,  name_policy='renumerate')
model.summary()
model.input.set_shape((1,) + model.input.shape[1:])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.experimental_new_converter=True
tflite_model = converter.convert()
with open('model_classify_no_coral.tflite', 'wb') as f:
    f.write(tflite_model)

# interpreter = tflite.Interpreter(model_path='model_classify.tflite')
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# interpreter.set_tensor(input_details[0]['index'], torch.Tensor(1, 150, 150,3))
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data.shape)
