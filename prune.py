import argparse
from models.experimental import attempt_download
from utils.torch_utils import model_info
from utils.general import make_divisible
import torch
import torch.nn as nn
import yaml
import ast
import numpy as np

def prune_single_layer(model, layer_index, percentage, criterion):
    next_layers = model.pruning_cfg[layer_index][0]
    concat_slices = model.pruning_cfg[layer_index][1]

    # get all modules with convolutional layers
    conv_modules = get_conv_layers(model)

    if layer_index < 0 or layer_index > len(conv_modules) - 1:
        raise Exception(f"layer {layer_index} index exceeded maximum number of layers: {len(conv_modules) - 1}")

    # current conv module that will be pruned
    # conv_module consists of a convolutional layer, batch normalization layer and SiLU activation
    conv_module = conv_modules[layer_index]

    # extract convolutional and batch normalization layers from the current conv_module
    # RepConv has two convolutional layers which is why conv_layers and bn_layers are lists
    conv_layers, bn_layers, _ = extract_layers_from_module(conv_module)

    num_params_before_pruning = sum([c.weight.numel() for c in conv_layers])

    device = conv_layers[0].weight.device
    c_out = conv_layers[0].out_channels # number of output channels of current layer

    pruned_indices = determine_pruned_indices(conv_layers, bn_layers, percentage, criterion)

    for c in conv_layers:
        new_weights = torch.index_select(c.weight, dim=0, index=pruned_indices).contiguous()
        c.weight = torch.nn.Parameter(new_weights)
        c.out_channels = len(pruned_indices)

    for b in bn_layers:
        new_weights = torch.index_select(b.weight, dim=0, index=pruned_indices).contiguous()
        new_bias = torch.index_select(b.bias, dim=0, index=pruned_indices).contiguous()
        new_running_var = torch.index_select(b.running_var, dim=0, index=pruned_indices).contiguous()
        new_running_mean = torch.index_select(b.running_mean, dim=0, index=pruned_indices).contiguous()

        b.num_features = len(pruned_indices)
        b.weight = torch.nn.Parameter(new_weights)
        b.running_var = torch.nn.Parameter(new_running_var).detach()
        b.running_mean = torch.nn.Parameter(new_running_mean).detach()
        b.bias = torch.nn.Parameter(new_bias)

    num_params_after_pruning_current_layer = sum([c.weight.numel() for c in conv_layers])

    print(
        f"\nLayer {layer_index}: Removed {c_out - len(pruned_indices)}/{c_out} filters based on criterion {criterion} and pruned {num_params_before_pruning - num_params_after_pruning_current_layer}/{num_params_before_pruning} ({round(100 * float(num_params_before_pruning - num_params_after_pruning_current_layer) / num_params_before_pruning, 2)}%) of the parameters in this convolutional layer. Pruning this layer affects the following layers:")

    # prune all layers that are affected by pruning the layer at index layer_index
    for next_layer, slice in zip(next_layers, concat_slices):
        offset = int(np.sum(model.pruning_cfg[next_layer][2][:slice]))
        if next_layer < 0 or next_layer > len(conv_modules) - 1:
            raise Exception(f"next_layer {next_layer} index exceeded maximum number of layers: {len(conv_modules) - 1}")
        next_conv_module = conv_modules[next_layer]
        next_conv_layers, _, next_implicits = extract_layers_from_module(next_conv_module)

        num_params_n = sum([next_conv.weight.numel() for next_conv in next_conv_layers])

        n_in = next_conv_layers[0].in_channels # number of input channels of the next layer

        # update slice sizes in pruning_cfg
        if len(model.pruning_cfg[next_layer][2]) > 0:
            model.pruning_cfg[next_layer][2][slice] -= c_out - len(pruned_indices)
        # indices need to be modified in case the next layer has concatenated input channels
        if offset > n_in-c_out:
            offset = n_in-c_out                         # fixed bug
        pruned_indices_next_layer = torch.cat(
            (torch.arange(offset).to(device), pruned_indices.to(device) + offset, torch.arange(offset + c_out, n_in).to(device)))
        for next_conv in next_conv_layers:
            new_weights = torch.index_select(next_conv.weight, dim=1, index=pruned_indices_next_layer).contiguous()
            next_conv.weight = torch.nn.Parameter(new_weights)
            next_conv.in_channels = n_in - c_out + len(pruned_indices)
        for next_implicit in next_implicits:
            new_implicit = torch.index_select(next_implicit.implicit, dim=1, index=pruned_indices_next_layer).contiguous()
            next_implicit.implicit = nn.Parameter(new_implicit)
            next_implicit.channel = n_in - c_out + len(pruned_indices)
        num_params_n_pruned = sum([next_conv.weight.numel() for next_conv in next_conv_layers])
        print(
            f"    Layer {next_layer}: Removed {n_in-len(pruned_indices_next_layer)}/{n_in} input channels and pruned {num_params_n - num_params_n_pruned}/{num_params_n} ({round(100 * float(num_params_n - num_params_n_pruned) / num_params_n, 2)}%) of the parameters in this convolutional layer.")

def extract_layers_from_module(conv_module):
    from models.common import Conv, RepConv
    conv_layers = []
    bn_layers = []
    implicits = []
    if isinstance(conv_module, Conv):
        conv_layers.append(conv_module.conv)
        bn_layers.append(conv_module.bn)
    elif isinstance(conv_module, RepConv):   # RepConv has two convolutional layers
        conv_layers.append(conv_module.rbr_dense[0])
        conv_layers.append(conv_module.rbr_1x1[0])
        bn_layers.append(conv_module.rbr_dense[1])
        bn_layers.append(conv_module.rbr_1x1[1])
    elif isinstance(conv_module, nn.Conv2d):
        conv_layers.append(conv_module)
    elif len(conv_module) > 1:
        conv_layers.append(conv_module[0])
        implicits.append(conv_module[1])
    return conv_layers, bn_layers, implicits

# get all modules with convolutional layers
def get_conv_layers(model):
    from models.common import Conv, RepConv
    from models.yolo import Detect, IDetect
    conv_layers = []
    for p in model.modules():
        if isinstance(p, Conv) or isinstance(p, RepConv):
            conv_layers.append(p)
        elif isinstance(p, Detect):
            for conv in p.m:
                conv_layers.append(conv)
        elif isinstance(p, IDetect):
            for conv, ia, im in zip(p.m, p.ia, p.im):
                conv_layers.append([conv, ia, im])
    return conv_layers

def determine_pruned_indices(conv, bn, percentage, criterion=0):
    device = conv[0].weight.device
    importances = torch.zeros_like(torch.norm(conv[0].weight, p=2, dim=[1, 2, 3]))
    if criterion == 0: # smallest L2
        for c in conv:  # loop in case conv has two convolutional layers (it usually has just one)
            importances += torch.norm(c.weight, p=2, dim=[1, 2, 3])
    elif criterion == 1: # largest L2
        for c in conv:
            importances -= torch.norm(c.weight, p=2, dim=[1, 2, 3])
    elif criterion == 2: # smallest L1
        for c in conv:
            importances += torch.norm(c.weight, p=1, dim=[1, 2, 3])
    elif criterion == 3: # largest L1
        for c in conv:
            importances -= torch.norm(c.weight, p=1, dim=[1, 2, 3])
    elif criterion == 4: # smallest bn scale factor
        for b in bn:
            importances += b.weight
    elif criterion == 5: # smallest bn scale factor * L1 norm
        for c, b in zip(conv, bn):
            importances += b.weight * torch.norm(c.weight, p=1, dim=[1, 2, 3])
    elif criterion == 6: # random
        importances += torch.rand(tensor.shape)

    indices = torch.argsort(importances)
    indices.to(device)
    n_to_prune = make_divisible(percentage * len(indices), 2)   # filters
    return torch.sort(indices[n_to_prune:])[0]

def num_params(model):
    a = 0
    for p in model.parameters():
        a += p.numel()
    return a

def prune_structured(model, pruning_params, criterion, tiny=False):
    from models.common import RepConv
    pruning_cfg = 'cfg/pruning_config/yolov7-tiny_pruning_cfg.yaml' if tiny else 'cfg/pruning_config/yolov7_pruning_cfg.yaml'
    if not hasattr(model, "pruning_cfg"):
        with open(pruning_cfg) as f:
            model.pruning_cfg = yaml.load(f, Loader=yaml.SafeLoader)['pruning_cfg']
    if type(pruning_params) == float and pruning_params < 1 and pruning_params > 0:
        num_conv_layers = len([conv for conv in model.modules() if isinstance(conv, nn.Conv2d)])
        pruning_params = [(i, pruning_params) for i in range(num_conv_layers)]
    pruning_params = sorted(pruning_params, key=lambda x: x[0])  # sorting is important to ensure that conv layers after concat layers are pruned from back to front
    print('Pruning model... ')
    num_params_before = num_params(model)
    conv_modules = get_conv_layers(model)

    for layer, amount in pruning_params:
        if isinstance(conv_modules[layer], RepConv):
            out = conv_modules[layer].rbr_dense[0].out_channels
        else:
            out = conv_modules[layer].conv.out_channels
        n_prune = make_divisible(amount * out, 2)
        if n_prune - out:
            prune_single_layer(model, layer, amount, criterion)
    num_params_after = num_params(model)
    print(
        f'\nPruned {num_params_before - num_params_after}/{num_params_before} parameters in total. Global sparsity: {round(100 * float(num_params_before - num_params_after) / num_params_before, 2)}%\n')
    model_info(model)

    return model

def load_pruned_model(weights, pruning_params, criterion, map_location=None, save=None):
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model = ckpt['ema' if ckpt.get('ema') else 'model']
    #model = model.float().fuse().eval())  # FP32 model
    pruned_model = prune_structured(model, pruning_params, criterion, tiny=True) if 'tiny' in weights[0] else\
        prune_structured(model, pruning_params, criterion)

    if save:
        ckpt['model'] = pruned_model
        torch.save(ckpt, save)

    return pruned_model.float().fuse().eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prune.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--pruning-params', type=str, default='', help='Pruning parameters can be defined as "[(l_1, p_1), (l_2, p_2), (l_3, p_3), ..., (l_n, p_n)]" where l_i are the indices of the layers to prune and p_i are the corresponding pruning rates for each layer')
    parser.add_argument('--criterion', type=int, default=0, help="Importance criterion for pruning:\n0= smallest L2-norm\n1= largest L2-norm\n2= smallest L1-norm\n3= largest L1-norm\n4= smallest batch normalization scale factor\n5= smallest batch normalization scale factor * L1-norm\n6= random")
    parser.add_argument('--name', default=None, help='save pruned model to name')
    opt = parser.parse_args()
    if len(opt.pruning_params) > 0:
        pruning_params_parsed = ast.literal_eval(opt.pruning_params)
    # python prune.py --weights yolov7.pt --pruning-params [] --criterion 0 --name yolov7-pruned.pt
    load_pruned_model(opt.weights, pruning_params_parsed, opt.criterion, save=opt.name)
