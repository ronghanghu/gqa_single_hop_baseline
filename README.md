# Single-Hop: A Simple Baseline for the GQA Dataset

This repository contains a simple model for the GQA dataset, using a single-hop attention over the visual features based on the encoded question, as specified in [`models_gqa/single_hop.py`](models_gqa/single_hop.py). It should serve as the baseline for more complicated models on the GQA dataset. The baseline is described as "single-hop" in the [LCGN paper](https://arxiv.org/pdf/1905.04405.pdf).

(If you are looking for the code for our LCGN model, please check out [this repo](https://github.com/ronghanghu/lcgn).)

It is applicable to three types of features from GQA:
* spatial features: `7 x 7 x 2048` ResNet spatial features (from the GQA dataset release)
* objects features: `N_{det} x 2048` Faster R-CNN ResNet object features (from the GQA dataset release)
* "perfect-sight" object names and attributes: obtained from the **ground-truth** scene graphs in GQA at **both training and test time**. This setting uses two one-hot vectors to represent each object's class name and attributes, and concatenate them as its visual feature. *It does not use the relation annotations in the scene graphs.*

It gets the following performance on the validation (`val_balanced`), the test-dev (`testdev_balanced`) and the test (`test_balanced`) split of the GQA dataset:

| Visual Feature Type  | Accuracy on `val_balanced` | Accuracy on `testdev_balanced` | Accuracy on `test_balanced` (obtained on EvalAI Phase: `test2019`) | Pre-trained model |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| spatial features | 54.97% | 48.61% | 49.11% | [download](https://people.eecs.berkeley.edu/~ronghang/projects/gqa_single_hop_baseline/pretrained_models/spatial/) |
| objects features | 62.02% | 53.78% | 54.42% | [download](https://people.eecs.berkeley.edu/~ronghang/projects/gqa_single_hop_baseline/pretrained_models/objects/) |
| "perfect-sight" object names and attributes | 86.98% | n/a* | n/a* | [download](https://people.eecs.berkeley.edu/~ronghang/projects/gqa_single_hop_baseline/pretrained_models/scene_graph/) |

*This setting requires using the GQA ground-truth scene graphs at both training and test time (only the object names and attributes are used; their relations are not used). Hence, it is not applicable to the test or the challenge setting.

## Installation

1. Install Python 3 (Anaconda recommended: https://www.continuum.io/downloads).
2. Install TensorFlow (we used TensorFlow 1.12.0 in our experiments):  
`pip install tensorflow-gpu`  (or `pip install tensorflow-gpu==1.12.0` to install TensorFlow 1.12.0)
3. Install a few other dependency packages (NumPy, HDF5, YAML):   
`pip install numpy h5py pyyaml`
3. Download this repository or clone with Git, and then enter the root directory of the repository:  
`git clone https://github.com/ronghanghu/gqa_single_hop_baseline.git && cd gqa_single_hop_baseline`

## Download the GQA dataset

Download the GQA dataset from https://cs.stanford.edu/people/dorarad/gqa/, and symbol link it to `exp_gqa/gqa_dataset`. After this step, the file structure should look like
```
exp_gqa/gqa_dataset
    questions/
        train_all_questions/
            train_all_questions_0.json
            ...
            train_all_questions_9.json
        train_balanced_questions.json
        val_all_questions.json
        val_balanced_questions.json
        submission_all_questions.json
        test_all_questions.json
        test_balanced_questions.json
    spatial/
        gqa_spatial_info.json
        gqa_spatial_0.h5
        ...
        gqa_spatial_15.h5
    objects/
        gqa_objects_info.json
        gqa_objects_0.h5
        ...
        gqa_objects_15.h5
    sceneGraphs/
        train_sceneGraphs.json
        val_sceneGraphs.json
    images/
        ...
```

Note that on GQA images are not needed for training or evaluation -- only questions, features and scene graphs (if you would like to run on the "perfect-sight" object names and attributes) are needed.

## Training on GQA

Note:  
* All of these three models are trained on the GQA `train_balanced` split, using a single GPU with a batch size of 128 for 100000 iterations, which takes a few hours on our machines.  
* By default, the above scripts use GPU 0. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2). During training, the script will write TensorBoard events to `exp_gqa/tb/{exp_name}/` and save the snapshots under `exp_gqa/tfmodel/{exp_name}/` (where `{exp_name}` is one of `spatial`, `objects` or `scene_graph`).

Pretrained models:  
* You may skip the training step and directly download the pre-trained models from the links at the top. The downloaded models should be saved under `exp_gqa/tfmodel/{exp_name}/` (where `{exp_name}` is one of `spatial`, `objects` or `scene_graph`).

Training steps：  

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Train with spatial features:  
`python exp_gqa/train.py --cfg exp_gqa/cfgs/spatial.yaml`
2. Train with objects features:  
`python exp_gqa/train.py --cfg exp_gqa/cfgs/objects.yaml`
3. Train with "perfect-sight" object names and attributes (one-hot embeddings):  
`python exp_gqa/train.py --cfg exp_gqa/cfgs/scene_graph.yaml`


## Testing on GQA

Note:
* The above evaluation script will print out the final VQA accuracy only on when testing on `val_balanced` or `testdev_balanced` split.   
* When running test on the `submission_all` split to generate the prediction file, **the displayed accuracy will be zero**, but the prediction file will be correctly generated under `exp_gqa/results/{exp_name}/` (where `{exp_name}` is one of `spatial`, `objects` or `scene_graph`), and the prediction file path will be displayed at the end. *It takes a long time to generate prediction files.*  
* By default, the above script uses GPU 0. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2).  

Testing steps:   

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Test with spatial features:  
    - test locally on the `val_balanced` split:   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/spatial.yaml TEST.SPLIT_VQA val_balanced`  
    - test locally on the `testdev_balanced` split:   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/spatial.yaml TEST.SPLIT_VQA testdev_balanced`  
    - generate the submission file on `submission_all` for EvalAI (this takes a long time):   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/spatial.yaml TEST.SPLIT_VQA submission_all`  
2. Test with objects features:  
    - test locally on the `val_balanced` split:   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/objects.yaml TEST.SPLIT_VQA val_balanced`  
    - test locally on the `testdev_balanced` split:   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/objects.yaml TEST.SPLIT_VQA testdev_balanced`  
    - generate the submission file on `submission_all` for EvalAI (this takes a long time):   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/objects.yaml TEST.SPLIT_VQA submission_all`  
3. Test with "perfect-sight" object names and attributes (one-hot embeddings):  
    - test locally on the `val_balanced` split:   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/scene_graph.yaml TEST.SPLIT_VQA val_balanced`  
    - test locally on the `testdev_balanced` split (**This won't work unless you have a file `testdev_sceneGraphs.json` under `exp_gqa/gqa_dataset/sceneGraphs/` that contains scene graphs for test-dev images**, which we don't):   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/scene_graph.yaml TEST.SPLIT_VQA testdev_balanced`  
    - generate the submission file on `submission_all` for EvalAI (**This won't work unless you have a file `submission_sceneGraphs.json` under `exp_gqa/gqa_dataset/sceneGraphs/` that contains scene graphs for all images**, which we don't):   
    `python exp_gqa/test.py --cfg exp_gqa/cfgs/scene_graph.yaml TEST.SPLIT_VQA submission_all`  

## Acknowledgements

The outline of the configuration code (such as `models_gqa/config.py`) is modified from the [Detectron](https://github.com/facebookresearch/Detectron) codebase.
