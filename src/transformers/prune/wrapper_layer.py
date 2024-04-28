import torch
from torch import nn

# Define WrapperLayer class
class WrapperLayer:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # number of features of the input=self.columns
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, input_X: torch.Tensor, output_X: torch.Tensor):
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