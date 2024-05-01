from .wrapper_layer import WrapperLayer
from .wrapper_layer import BloomWrapperLayer, LlamaWrapperLayer
from .sparsity_util import find_layers
import torch
from torch import nn
import os

def prune_by_weight_importance(
        weight: torch.tensor,
        weight_importance: torch.tensor,
        pruning_percentage,
        layer_wise_weight_selection=False) -> torch.tensor:
    original_size = weight.size()
    topk = int(pruning_percentage * weight_importance.reduce())
    preserved_indices = weight.view(-1,).topk(topk).indices
    mask = torch.zeros_like(weight).scatter(0, preserved_indices, 1).bool().half()
    return weight.view(original_size) * mask

def prune_by_column_importance(weight: torch.tensor, weight_importance: torch.tensor) -> torch.tensor:
    pass

PRUNING_FUNC_MAP = {
    'weight', prune_by_weight_importance,
    'column', prune_by_column_importance
}

#PruneMetadata is used to store the statistics during the forward pass of the model.
class PruneMetadata:
    def __init__(self, model, output_path=None, enable_weight_wise_pruning=True, pruning_strategy='weight'):
        self.all_wrapper_layers = []
        self.handles = []
        self.model = model
        self.output_path = output_path
        self.enable_weight_wise_pruning = enable_weight_wise_pruning
        self.sparsity_percentage = 1.0
        assert pruning_strategy in PRUNING_FUNC_MAP
        self.pruning_func = PRUNING_FUNC_MAP[pruning_strategy]

    def register_hooks_for_layers(self, layers):
        for id, layer in enumerate(layers):
            subset = self.find_instrument_layers(layer)
        
            # Wrapper layer is used to record the statistics of each layer
            wrapper_layers = {}
            for name in subset:
                wrapper_layers[name] = self.create_wrapper_layer(subset[name], layer_id=id, layer_name=name)
            self.all_wrapper_layers.append(wrapper_layers)
            
            def add_batch(layer_id, name, wrapper_layer):
                def tmp(_, inp, out):
                    # print('[DEBUG-0]layer_id:{}, layer_name:{}'.format(layer_id, name))
                    wrapper_layer.add_batch(inp[0].data, out.data)
                    # print('[DEBUG-1]layer_id:{}, layer_name:{}'.format(layer_id, name))
                return tmp

            for name, wrapper_layer in wrapper_layers.items():
                if self.enable_weight_wise_pruning:
                    module = subset[name]
                    # Load the activations from previously recorded files
                    activation_info = torch.load(os.path.join(self.output_path, f"{id}_{name}.pt"))
                    # prune weight based on recorded activation information
                    module.weight = nn.Parameter(self.pruning_func(module.weight, activation_info, self.sparsity_percentage))
                else:
                    # record activation information
                    self.handles.append(subset[name].register_forward_hook(add_batch(id, name, wrapper_layer)))

    def find_instrument_layers(self, layer):
        return find_layers(layer)
    
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return WrapperLayer(layer, layer_id, layer_name)

    def print(self, save_weight_importance=True):
        print("PruneMetadata")
        print("all_wrapper_layers:")
        for id, wrapper_layers in enumerate(self.all_wrapper_layers):
            print(" layer_id:", id)
            for name, wrapper_layer in wrapper_layers.items():
                print("  layer_name:", name)
                print("    rows:", wrapper_layer.rows)
                print("    columns:", wrapper_layer.columns)
                print("    nsamples:", wrapper_layer.nsamples)
                print("    scaler_row.shape:", wrapper_layer.scaler_row.shape)
                weight_importance = wrapper_layer.get_weight_importance()
                print("    weight_importance.shape:", weight_importance.shape)
                if self.output_path is not None:
                    if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)
                    filename = f"{id}_{name}.pt"
                    torch.save(weight_importance, os.path.join(self.output_path, filename))
        
# TODO: implement this
class BloomPruneMetadata(PruneMetadata):
    def __init__(self, model, output_path):
        super().__init__(model, output_path)
        self.output_path = output_path
        
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return BloomWrapperLayer(layer, layer_id, layer_name)
    
class LlamaPruneMetadata(PruneMetadata):
    def __init__(self, model, activation_func, output_path):
        super().__init__(model, output_path)
        self.activation_func = activation_func
        
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return LlamaWrapperLayer(layer, layer_id, layer_name, self.activation_func)