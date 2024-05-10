from .wrapper_layer import WrapperLayer
from .wrapper_layer import BloomWrapperLayer, LlamaWrapperLayer, OPTWrapperLayer
from .sparsity_util import find_layers, PRUNING_FUNC_MAP, RECORDED_TASKS
import torch
from torch import nn
import os
from transformers.utils import logging
logger = logging.get_logger(__name__)

# PruneMetadata is used to record the statistics during the forward pass of the model.
# It can also prune the model based on recorded statistics 
class PruneMetadata:
    def __init__(self, model, config):
        self.all_wrapper_layers = []
        self.handles = []
        self.model = model
        self.output_path = config.output_path
        self.enable_weight_activation_based_pruning = config.enable_weight_activation_based_pruning
        self.sparsity_ratio = config.sparsity_ratio
        self.target_model = config.target_model
        assert config.pruning_strategy in PRUNING_FUNC_MAP
        self.pruning_func = PRUNING_FUNC_MAP[config.pruning_strategy]
        self.task_angostic_pruning = config.task_angostic_pruning
        self.record_weight_wise_activation = config.record_weight_wise_activation
        self.layers = None
        self.total_num_weights = 0
        self.num_weights_pruned = 0
        self.analyze_layer_norm_affect = config.analyze_layer_norm_affect

    def set_instrumented_layers(self, layers):
        self.layers = layers
        
    def instrument_layers(self, layers):
        assert self.layers != None
        for id, layer in enumerate(layers):
            if self.analyze_layer_norm_affect:
                if self.target_model == "llama":
                    from transformers.models.llama.modeling_llama import LlamaRMSNorm
                    subset = find_layers(layer, layers=[LlamaRMSNorm])
                elif self.target_model == "opt":
                    subset = find_layers(layer, layers=[nn.LayerNorm])
            else:
                subset = find_layers(layer)
        
            # Wrapper layer is used to record the statistics of each layer
            wrapper_layers = {}
            for name in subset:
                wrapper_layers[name] = self.create_wrapper_layer(subset[name], layer_id=id, layer_name=name)
            self.all_wrapper_layers.append(wrapper_layers)
            
            for name, wrapper_layer in wrapper_layers.items():
                module = subset[name]
                if self.enable_weight_activation_based_pruning:
                    logger.warn("Pruning weight matrix for {}_{}".format(wrapper_layer.layer_id, name))
                    # prune weight based on recorded activation information
                    pruned_weight, pruned_percentage = self.pruning_func(
                            module.weight.data, 
                            self.load_weight_activations(wrapper_layer.layer_id, name), 
                            self.sparsity_ratio)
                    logger.warn(f"{pruned_percentage:.2f}" + 
                                "% of the least important " + 
                                str(100 * self.sparsity_ratio) + 
                                "% weights are common among all tasks, and zeroed out.")
                    num_weights = pruned_weight.view(-1,).shape[0]
                    self.total_num_weights += num_weights
                    self.num_weights_pruned += (pruned_percentage / 100 * self.sparsity_ratio * num_weights)
                    module.weight.data = nn.Parameter(pruned_weight)
                elif self.record_weight_wise_activation:
                    # record activation information
                    def add_batch(layer_id, name, wrapper_layer):
                        def tmp(_, inputs, output):
                            # print('[DEBUG-0]layer_id:{}, layer_name:{}'.format(layer_id, name))
                            wrapper_layer.add_batch(inputs[0].data, output.data)
                            # print('[DEBUG-1]layer_id:{}, layer_name:{}'.format(layer_id, name))
                        return tmp
                    self.handles.append(module\
                        .register_forward_hook(add_batch(id, name, wrapper_layer)))
                elif self.analyze_layer_norm_affect:
                    def record_in_out(layer_id, name, wrapper_layer):
                        def tmp(_, inputs, output):
                            # print('[DEBUG-0]layer_id:{}, layer_name:{}'.format(layer_id, name))
                            wrapper_layer.record_in_out(inputs[0].data, output.data)
                            # print('[DEBUG-1]layer_id:{}, layer_name:{}'.format(layer_id, name))
                        return tmp
                    self.handles.append(module.register_forward_hook(
                        record_in_out(id, name, wrapper_layer)
                    ))

    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return WrapperLayer(layer, layer_id, layer_name)

    def load_weight_activations(self, layer_id, layer_name):
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
                            base_recorded_statistics_folder,
                            task,
                            f"{layer_id}_{layer_name}.pt")))
        else:
            activation_infos.append(
                torch.load(
                    os.path.join(
                        self.output_path, 
                        f"{layer_id}_{layer_name}.pt")))
        return activation_infos

    def print_statistics(self, save_weight_importance=True):
        logger.warn("PruneMetadata")
        logger.warn("For all layers:")
        if self.record_weight_wise_activation or self.analyze_layer_norm_affect:
            for id, wrapper_layers in enumerate(self.all_wrapper_layers):
                logger.warn(f" layer_id:{id}")
                for name, wrapper_layer in wrapper_layers.items():
                    logger.warn(f"  layer_name:{name}")
                    if self.analyze_layer_norm_affect:
                        numbers = wrapper_layer.sims
                        average = sum(numbers) / len(numbers)
                        logger.warn(f"    average cosine sim of layer norm: {average.item()}")
                        continue
                    logger.warn("    rows:", wrapper_layer.rows)
                    logger.warn("    columns:", wrapper_layer.columns)
                    logger.warn("    nsamples:", wrapper_layer.nsamples)
                    logger.warn("    scaler_row.shape:", wrapper_layer.scaler_row.shape)
                    weight_importance = wrapper_layer.get_weight_importance()
                    logger.warn("    weight_importance.shape:", weight_importance.shape)
                    if self.output_path is not None and save_weight_importance:
                        if not os.path.exists(self.output_path):
                            os.makedirs(self.output_path)
                        filename = f"{id}_{name}.pt"
                        torch.save(weight_importance, os.path.join(self.output_path, filename))
        if self.enable_weight_activation_based_pruning:
            logger.warn(f"{(100 * self.num_weights_pruned / self.total_num_weights):.2f}" +
                        "% weights in the entire model are pruned.")

class BloomPruneMetadata(PruneMetadata):
    def __init__(self, model, config):
        super().__init__(model, config)
        
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return BloomWrapperLayer(layer, layer_id, layer_name)

class LlamaPruneMetadata(PruneMetadata):
    def __init__(self, model, activation_func, config):
        super().__init__(model, config)
        self.activation_func = activation_func
        
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return LlamaWrapperLayer(layer, layer_id, layer_name, self.activation_func)

class OPTPruneMetadata(PruneMetadata):
    def __init__(self, model, activation_func, config):
        super().__init__(model, config)
        self.activation_func = activation_func

    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return OPTWrapperLayer(layer, layer_id, layer_name, self.activation_func)