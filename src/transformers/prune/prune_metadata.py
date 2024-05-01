from .wrapper_layer import WrapperLayer
from .wrapper_layer import BloomWrapperLayer, LlamaWrapperLayer
from .sparsity_util import find_layers, PRUNING_FUNC_MAP, RECORDED_TASKS
import torch
from torch import nn
import os

# PruneMetadata is used to record the statistics during the forward pass of the model.
# It can also prune the model based on recorded statistics 
class PruneMetadata:
    def __init__(self, model, output_path=None, enable_weight_wise_pruning=True, pruning_strategy='weight', sparsity_ratio=1.0, task_angostic_pruning=False):
        self.all_wrapper_layers = []
        self.handles = []
        self.model = model
        self.output_path = output_path
        self.enable_weight_wise_pruning = enable_weight_wise_pruning
        self.sparsity_ratio = sparsity_ratio
        assert pruning_strategy in PRUNING_FUNC_MAP
        self.pruning_func = PRUNING_FUNC_MAP[pruning_strategy]
        self.task_angostic_pruning = task_angostic_pruning

    def register_hooks_for_layers(self, layers):
        for id, layer in enumerate(layers):
            subset = self.find_instrument_layers(layer)
        
            # Wrapper layer is used to record the statistics of each layer
            wrapper_layers = {}
            for name in subset:
                wrapper_layers[name] = self.create_wrapper_layer(subset[name], layer_id=id, layer_name=name)
            self.all_wrapper_layers.append(wrapper_layers)
            
            def add_batch(layer_id, name, wrapper_layer):
                def tmp(_, inputs, output):
                    # print('[DEBUG-0]layer_id:{}, layer_name:{}'.format(layer_id, name))
                    wrapper_layer.add_batch(inputs[0].data, output.data)
                    # print('[DEBUG-1]layer_id:{}, layer_name:{}'.format(layer_id, name))
                return tmp

            for name, wrapper_layer in wrapper_layers.items():
                if self.enable_weight_wise_pruning:
                    module = subset[name]
                    activation_infos = []
                    # Load the activations from previously recorded files
                    if self.task_angostic_pruning:
                        # Load all previously recorded activation infomation, only prune those that does not significantly effect all tasks.
                        # NOTICE: change the `RECORDED_TASKS` based on what have recorded
                        # We assume all recorded statistics are orginized in the same folder
                        base_recorded_statistics_folder = self.output_path[:self.output_path.rfind(os.path.sep)]
                        for task in RECORDED_TASKS:
                            activation_infos.append(
                                torch.load(
                                    os.path.join(
                                        self.output_path,
                                        base_recorded_statistics_folder,
                                        f"{id}_{task}.pt")))
                    else:
                        activation_infos.append(
                            torch.load(
                                os.path.join(
                                    self.output_path, 
                                    f"{id}_{name}.pt")))
                    # prune weight based on recorded activation information
                    module.weight = nn.Parameter(
                        self.pruning_func(module.weight, activation_infos, self.sparsity_percentage))
                else:
                    # record activation information
                    self.handles.append(subset[name]\
                        .register_forward_hook(add_batch(id, name, wrapper_layer)))

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
                if self.output_path is not None and save_weight_importance:
                    if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)
                    filename = f"{id}_{name}.pt"
                    torch.save(weight_importance, os.path.join(self.output_path, filename))
        
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