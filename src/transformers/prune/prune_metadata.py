from .wrapper_layer import WrapperLayer
from .sparsity_util import find_layers
import torch

#PruneMetadata is used to store the statistics during the forward pass of the model.
class PruneMetadata:
    def __init__(self, output_path=None):
        self.all_wrapper_layers = []
        self.handles = []
        self.output_path = output_path
    
    def register_hooks_for_layers(self, layers):
        for id, layer in enumerate(layers):
            subset = find_layers(layer)
        
            # Wrapper layer is used to record the statistics of each layer
            wrapper_layers = {}
            for name in subset:
                wrapper_layers[name] = WrapperLayer(subset[name], layer_id=id, layer_name=name)
            self.all_wrapper_layers.append(wrapper_layers)
            
            def add_batch(layer_id, name, wrapper_layer):
                def tmp(_, inp, out):
                    # print('[DEBUG-0]layer_id:{}, layer_name:{}'.format(layer_id, name))
                    wrapper_layer.add_batch(inp[0].data, out.data)
                    # print('[DEBUG-1]layer_id:{}, layer_name:{}'.format(layer_id, name))
                return tmp
            for name, wrapper_layer in wrapper_layers.items():
                self.handles.append(subset[name].register_forward_hook(add_batch(id, name, wrapper_layer)))
        
    def print(self):
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
                if output_path is not None:
                    #TODO(YCW): make the path configurable
                    filename = f"{id}_{name}.pt"
                    torch.save(weight_importance, output_path + '/' + filename)
        
        print('hook_size:', len(self.handles))