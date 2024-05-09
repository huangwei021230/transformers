import torch
from torch import nn
import torch.nn.functional as F
# Define WrapperLayer class
class WrapperLayer:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        assert layer != None
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1] if len(layer.weight.data.shape) > 1 else None

        # number of features of the input=self.columns
        self.scaler_row = None
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name
        self.layer_activation = None
        self.sims = []
        
    def record_in_out(self, input_X: torch.Tensor, output_X: torch.Tensor):
        assert input_X.shape == output_X.shape
        self.sims.append(F.cosine_similarity(input_X, output_X).mean())

    def add_batch(self, input_X: torch.Tensor, output_X: torch.Tensor):
        if self.scaler_row is None:
            # keep the layer weight and scale_row always in same device
            self.dev = self.layer.weight.device
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
            
        if len(input_X.shape) == 2:
            input_X = input_X.unsqueeze(0)
        batch_size = input_X.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(input_X.shape) == 3:
                input_X = input_X.reshape((-1, input_X.shape[-1]))
            input_X = input_X.t()

        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size        
        input_X = input_X.type(torch.float32)
        self.scaler_row += torch.norm(input_X, p=2, dim=1) ** 2  / self.nsamples
        # print('[DEBUG]layer_id:{}, layer_name:{}, nsamples:{}'.format(self.layer_id, self.layer_name, self.nsamples))
    
    def get_weight_importance(self):
        result = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1,-1)))
        if self.layer_activation != None:
            return self.layer_activation(result).half()
        return result.half()

class BloomWrapperLayer(WrapperLayer):
    def __init__(self, layer, layer_id=0, layer_name="none"):
        super().__init__(layer, layer_id, layer_name)
        from transformers.models.bloom.modeling_bloom import BloomGelu
        if layer_name == 'mlp.dense_h_to_4h':
            self.layer_activation = BloomGelu()

class LlamaWrapperLayer(WrapperLayer):
    def __init__(self, layer, layer_id=0, layer_name="none", activation_func=None):
        super().__init__(layer, layer_id, layer_name)
        # TODO: inject activation function for specific layers
        if layer_name == 'mlp.gate_proj':
            self.layer_activation = activation_func