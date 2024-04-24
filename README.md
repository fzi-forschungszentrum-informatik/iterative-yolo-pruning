# Iterative Filter Pruning for Concatenation-Based CNN Architectures
Repository of the paper **Iterative Filter Pruning for Concatenation-based
CNN Architectures** at [IJCNN 2024](https://2024.ieeewcci.org/). 

 We propose a method to handle concatenation layers, based on the connectivity graph of convolutional layers. By automating iterative sensitivity analysis, pruning, and subsequent model fine-tuning, we can significantly reduce model size both in terms of the number of parameters and FLOPs, while keeping comparable model accuracy. Finally, we deploy pruned models to FPGA and NVIDIA Jetson Xavier AGX.  Pruned models demonstrate a 2x speedup for the convolutional layers in comparison to the unpruned counterparts and reach real-time capability with 14 FPS on FPGA.

With the provided scripts, you can run the proposed iterative filter pruning approach exemplary for YOLOv7. The code builds upon [YOLOv7, as implemented by Chien-Yao Wang](https://github.com/WongKinYiu/yolov7/) and is licensed under GPL3.0, see [here](./LICENSE.md) 
 
## Structure
```
.
├── README.md
├── pruning_config - folder containing pruning configuration file for yolov7 and yolov7-tiny networks
├── tools
│  ├── Analysis.ipynb - tool for anaylsing the results of pruned models
│  ├── change_yolo_activation.ipynb - tool for saving the yolov7-tiny model with different activation functions
│  └── compare_map_per_class.ipynb - tool for the comparison of mAP results for the certain classes of pruned models
├── layer_selection.py - script for the selection of layers based on the sensitivity analysis 
├── prune.py - script for the pruning of models which saves a pruned model if executed
├── requirements.txt - equivalent to the requirements in YOLOv7
├── test.py - script for testing the models and for the sensitivity analysis (modified test.py of YOLOv7)
└── train.py - script for the fine-tuning of the pruned models (modified train.py of YOLOv7)
```

## Setup
1. Clone YOLOv7 from https://github.com/WongKinYiu/yolov7/ and add path to it to your PYTHONPATH. Navigate to the repo and replace the scripts: `test.py`, `train.py` with the ones provided here and add the scripts `prune.py` and `layer_selection.py` to the repo.
2. Add the `pruning` folder that includes the pruning configuration files of `yolov7.pt` and `yolov7-tiny.pt` into the `cfg` in the repo. The pruning configuration file which shows how convolutional layers of a model connected is used to correctly identify and remove the slice of a layer affected by pruning.

## Iterative Pruning
The steps explained below can be repeated after fine-tuning meaning that the YOLOv7 models can be iteratively pruned. Pruning iterations of `yolov7-tiny.pt` should include `tiny` in their name so the correct pruning configuration file is selected for the next iteration of pruning.

### Testing
Testing YOLOv7 model on the validation set.
``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```
We add new options to modify YOLOv7:

| Option        | Description                                |
|---------------|--------------------------------------------|
| `--modification`   | `pruned-structured`|
| `--pruning-params`   | Can either be a float value between `0.0` and `1.0` to indicate the pruning ratio that is applied to all layers or a list of tuples specifying the pruning ratio for each layer `"[(2, 0.5), (6, 0.5), (10, 0.3)]"`             |
| `--criterion` | Importance criteria for selecting filters to prune in structured pruning: <br>- `0` smallest L2-norm<br>- `1` largest L2-norm<br>- `2` smallest L1-norm<br>- `3` largest L1-norm<br>- `4` smallest batch normalization scale factor<br>- `5` smallest batch normalization scale factor * L1-norm<br>- `6` random |

### Sensitivity Analysis
To conduct a sensitivity analysis for pruning individual layers of YOLOv7, the task must be set to `--task pruning_sensitivity_analysis`. It can be applied to structured pruning with `--modification prune-structured`. The outputs are written to text files specified with `--prune-output output.txt` for the pruning rates with `--pruning-rate '[0.25, 0.5, 0.75]'` and saved in `output/yolov7_training` directory:
``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --task pruning_sensitivity_analysis --device 0 --weights yolov7_training.pt --name yolov7_640_sensitivity --modification prune-structured --prune-output output.txt --pruning-rate '[0.25, 0.5, 0.75]'
```
`output_{pruning_rate}.txt` will contain a list where each entry corresponds to the result of pruning each layer with the specified pruning rate individually. The entries are tuples containing the following information:
- (layer_index, metrics, timing)
  - layer_index: Index of the layer
  - metrics:
    - Mean Recall value
    - Mean Precision value
    - Mean Average Precision at IoU 0.50 (mAP@.50)
    - Mean Average Precision from IoU 0.50 to 0.95 (mAP@.50:.95)
    - List of mAP values for all classes
  - timing:
    - Time taken for inference
    - Time taken for Non-Maximum Suppression (NMS)
    - Combined time for inference and NMS
    - Height of the image
    - Width of the image
    - Batch size used
  - Number of parameters
  - Number of GFLOPS

### Selecting layers and pruning rates
Once the sensitivity analysis completed the pruning parameters are selected by using the command below. The command will print the list of pruning parameters in the terminal logs.
``` shell
python layer_selection.py --output output/yolov7_training --params 36907898 --flops 104.514 --params-layers 6 --flops-layers 5 --params-map 20 --flops-map 20
```

| Option            | Description                                                                           |
|-------------------|---------------------------------------------------------------------------------------|
| `--output`        | output directory of sensitivity analysis                                              |
| `--params`        | number of parameters of a YOLOV7 model (can be obtained by testing the model)         |
| `--flops`         | number of GFLOPS of a YOLOV7 model (can be obtained by testing the model)             |
| `--params-layers` | number of layers to be pruned for parameters                                          |
| `--flops-layers`  | number of layers to be pruned for FLOPS                                               |
| `--params-map`    | impact of map degradation on the selection of layers and pruning rates for parameters |
| `--flops-map`     | impact of map degradation on the selection of layers and pruning rates for FLOPS      |

### Testing with the pruning parameters
Testing YOLOv7 model by pruning with the selected pruning parameters is used to determine how many epochs are needed for fine-tuning:
``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7_training.pt --name yolov7_640_val --modification prune-structured --pruning-params '[(8, 0.5), (18, 0.5), (28, 0.75), (29, 0.75), (32, 0.75), (33, 0.75), (41, 0.5), (43, 0.5), (45, 0.5), (46, 0.5)]'
```

### Fine-tuning with the pruning parameters
Fine-tuning and pruning can be simultaneously applied to YOLOv7 models by specifying the pruning parameters and the hyperparameters for fine-tuning in `data/hyp.scratch.custom.yaml` using the following command:
``` shell
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --weights "yolov7_training.pt" --name yolov7-fine-tuned.pt --hyp data/hyp.scratch.custom.yaml --pruning-params "[(8, 0.5), (18, 0.5), (28, 0.75), (29, 0.75), (32, 0.75), (33, 0.75), (41, 0.5), (43, 0.5), (45, 0.5), (46, 0.5)]" --criterion 0
```

# Citation

If you find this code useful for your research, please cite our paper:

```latex
@InProceedings{pavlitska2024iterative,
  author    = {Pavlitska, Svetlana and Bagge, Oliver and Peccia, Federico and Mammadov, Toghrul and Zöllner, J. Marius},
  title     = {{Iterative Filter Pruning for Concatenation-based CNN Architectures}},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2024}
}
```

