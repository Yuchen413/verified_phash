# import sys
# sys.path.insert(0,'/verified_phash')

import onnx
import torch
from NeuralHash_track_is_true import NeuralHash_track_is_true

onnx_model = onnx.load('model.onnx')
pytorch_model = NeuralHash_track_is_true(onnx_model)
torch.save(pytorch_model.state_dict(), '/home/yuchen/code/verified_phash/AppleNeuralHash2ONNX/nerualhash/model.pth')