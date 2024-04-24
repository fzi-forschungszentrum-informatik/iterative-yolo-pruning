import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def select(output, parameters, flops, params_layers, flops_layers, params_map, flops_map, save=False):
    rates = []
    layer = []
    precision = []
    params = []
    flps = []
    params_val = []
    flps_val = []
    conv_layers = 55 if 'tiny' in output else 89

    def clean(line, char):
        a = []
        lis = line.split(',')
        lis.pop()
        for i in range(len(lis)):
            for c in char:
                lis[i] = lis[i].replace(c, '')
            a.append(float(lis[i]))
        return a

    for idx, file in enumerate(sorted(os.listdir(output))):
        name = Path(file).stem
        rate = name.split('_')
        if len(rate) != 2 or not rate[1].isdigit() or rate[0] == 'test':
            continue
        rates.append(int(rate[1])/100)
        layer.append([])
        precision.append([])
        params.append([])
        flps.append([])
        params_val.append([])
        flps_val.append([])
        with open(os.path.join(output, file),   'r') as f:
            for line in f:
                if len(line) > 2:
                    a = clean(line, char='() []')
                    layer[-1].append(a[0])
                    precision[-1].append(a[4])
                    params[-1].append(parameters - a[-2])
                    flps[-1].append(flops - a[-1])

    for idx, lay in enumerate(layer):
        if len(lay) < conv_layers:
            raise Exception(f'Sensitivity analysis for {rates[idx]} is incomplete')
        elif len(lay) > conv_layers:
            raise Exception(f'Sensitivity analysis for {rates[idx]} has more conv layers than expected')

    # Parameters selection
    for i in range(len(params_val)):
        for idx, param in enumerate(params[i]):
            params_val[i].append(param * (precision[i][idx] ** params_map))

    max_p = max([max(params_val[i]) for i in range(len(params_val))])
    transposed = [list(i) for i in zip(*params_val)]
    max_val = [max(transposed[i]) for i in range(len(transposed))]
    max_rate = [np.argmax(transposed[i]) for i in range(len(transposed))]
    frac = 2
    while True:
        #print(frac)
        number_params = [max_val.index(val) for val in max_val if val > max_p / frac]
        if len(number_params) < params_layers-1:
            frac += 0.001
        elif len(number_params) > params_layers:
            frac -= 0.001
        else:
            break
    rate_params = [max_rate[num] for num in number_params]

    # FLOPS selection
    for i in range(len(flps_val)):
        for idx, flp in enumerate(flps[i]):
            flps_val[i].append(flp * (precision[i][idx] ** flops_map))

    max_p = max([max(flps_val[i]) for i in range(len(flps_val))])
    transposed = [list(i) for i in zip(*flps_val)]
    max_val = [max(transposed[i]) for i in range(len(transposed))]
    max_rate = [np.argmax(transposed[i]) for i in range(len(transposed))]
    frac = 2
    while True:
        number_flps = [max_val.index(val) for val in max_val if val > max_p / frac]
        if len(number_flps) < flops_layers-1:
            frac += 0.001
        elif len(number_flps) > flops_layers:
            frac -= 0.001
        else:
            break
    rate_flps = [max_rate[num] for num in number_flps]

    # selected layers and pruning rates:
    layers = {}
    for num, r in zip(number_flps, rate_flps):
        layers[(num, rates[r])] = 'FLOPS'
    for num, r in zip(number_params, rate_params):
        if (num, rates[r]) in layers:
            layers[(num, rates[r])] = 'both'
        else:
            layers[(num, rates[r])] = 'params'
    print(f'Layers and pruning rates in {list(layers.keys())} '
          f'\nare selected for values in {list(layers.values())} respectively')

    # Saving SA diagrams
    if save:
        folder = os.path.join('graphs', f'SA_{Path(output).name}')
        os.makedirs(folder, exist_ok=True)

        # Sensitivity analysis for parameters
        fig1, ax1 = plt.subplots()
        hans1 = []
        labs1 = []
        for idx in range(len(layer)):
            hans1.append(plt.plot(layer[idx], [p / 1E6 for p in params[idx]]))
            labs1.append('pruning rate: ' + str((idx + 1) * 25) + '%')

        hans1 = [h[0] for h in hans1]

        for idx, val in enumerate(number_params):
            dots = ax1.plot(val, params[rate_params[idx]][val] / 1E6, ".", color='limegreen', ms=8, mec='k', mew=0.8)
        hans1.append(dots[0])
        labs1.append('Selected rates for\npruning parameters')
        ax1.legend(hans1, labs1)

        ax1.set_xlabel('Convolutional Layer Index')
        ax1.set_ylabel('Pruned Parameters (millions)')

        ax1.locator_params(axis='both', nbins=10)
        ax1.grid(True)
        fig1.set_size_inches(8, 6)
        fig1.savefig(os.path.join(folder, 'sa_parameters.pdf'), bbox_inches='tight', transparent=True, pad_inches=0)

        # Sensitivity analysis for FLOPS
        fig2, ax2 = plt.subplots()
        hans2 = []
        labs2 = []

        for idx in range(len(layer)):
            hans2.append(plt.plot(layer[idx], flps[idx]))
            labs2.append('pruning rate: ' + str((idx + 1) * 25) + '%')

        hans2 = [h[0] for h in hans2]

        for idx, val in enumerate(number_flps):
            dots = ax2.plot(val, flps[rate_flps[idx]][val], ".", color="royalblue", ms=8, mec='k', mew=0.8)
        hans2.append(dots[0])
        labs2.append('Selected rates for\npruning FLOPS')
        ax2.legend(hans2, labs2)

        ax2.set_xlabel('Convolutional Layer Index')
        ax2.set_ylabel('Pruned GFLOPS')

        ax2.locator_params(axis='both', nbins=10)
        ax2.grid(True)
        fig2.set_size_inches(8, 6)
        fig2.savefig(os.path.join(folder, 'sa_flops.pdf'), bbox_inches='tight', transparent=True, pad_inches=0)

        # Sensitivity analysis for map
        fig0, ax0 = plt.subplots()
        hans0 = []
        labs0 = []

        for idx in range(len(layer)):
            hans0.append(plt.plot(layer[idx], precision[idx]))
            labs0.append('pruning rate: ' + str((idx + 1) * 25) + '%')

        hans0 = [h[0] for h in hans0]

        for val in layers:
            h = ax0.plot(val[0], precision[int(val[1] / 0.25 - 1)][val[0]], ".w", ms=8, mec='k', mew=0.8)
            if layers[val] == 'FLOPS':
                ax0.plot(val[0], precision[int(val[1] / 0.25 - 1)][val[0]], ".", color='royalblue', ms=8, mec='k', mew=0.8)
            elif layers[val] == 'both':
                ax0.plot(val[0], precision[int(val[1] / 0.25 - 1)][val[0]], ".r", ms=8, mec='k', mew=0.8)
            elif layers[val] == 'params':
                ax0.plot(val[0], precision[int(val[1] / 0.25 - 1)][val[0]], ".", color='limegreen', ms=8, mec='k', mew=0.8)
        hans0.append(h[0])
        labs0.append('Selected pruning rates\n(red: for both)')
        ax0.legend(hans0, labs0)

        ax0.set_xlabel('Convolutional Layer Index')
        ax0.set_ylabel('mAP')

        ax0.locator_params(axis='both', nbins=10)
        ax0.grid(True)
        fig0.set_size_inches(8, 6)
        fig0.savefig(os.path.join(folder, 'sa_map.pdf'), bbox_inches='tight', transparent=True, pad_inches=0)

        print(f'Sensitivity analysis graphs are saved in {folder}')

    return str(list(layers.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='layer_selection.py')
    parser.add_argument('--output', type=str, default='output/yolov7_training',
                        help='directory containing text files which are outputs of sensitivity analysis')
    parser.add_argument('--params', type=int, default=36907898, help='number of parameters')
    parser.add_argument('--flops', type=float, default=104.514, help='number of FLOPS')
    parser.add_argument('--params-layers', type=int, default=6, help='number of layers to be pruned for parameters')
    parser.add_argument('--flops-layers', type=int, default=5, help='number of layers to be pruned for FLOPS')
    parser.add_argument('--params-map', type=float, default=20, help='impact of map on parameter selection')
    parser.add_argument('--flops-map', type=float, default=20, help='impact of map on FLOPS selection')
    parser.add_argument('--save', action='store_true', help='saving sensitivity analysis graphs')
    opt = parser.parse_args()

    layers = \
        select(opt.output, opt.params, opt.flops, opt.params_layers, opt.flops_layers, opt.params_map,
               opt.flops_map, opt.save)

