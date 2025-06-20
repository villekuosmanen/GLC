# Getting Started with PySlowFast

This document provides a brief intro of launching jobs in GLC for training and testing. Before launching any job, make sure you have properly installed the GLC following the instruction in [README.md](README.md) and you have prepared the dataset following [DATASET.md](slowfast/datasets/DATASET.md) with the correct format.

## Train from Kinetics Pretrained Models

Here we can start with training a model by running:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py \
 --init_method tcp://localhost:9877 \
 --cfg configs/Egtea/MVIT_B_16x4_CONV.yaml \
 TRAIN.BATCH_SIZE 16 \
 TEST.ENABLE False \
 NUM_GPUS 4 \
 TRAIN.CHECKPOINT_FILE_PATH /path/to/pretrained/K400_MVIT_B_16x4_CONV.pyth \
 OUTPUT_DIR checkpoints/GLC
 DATA.PATH_PREFIX /path/to/Ego4D/clips.gaze
```

[//]: # (You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or )

You can simply add

```
DATA:
  PATH_PREFIX: path_to_your_dataset
```
To the yaml configs file, then you do not need to pass it to the command line every time.


## Resume from an Existing Checkpoint

If your checkpoint is trained by PyTorch, then you can add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
```

## Perform Test
We have `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for the current job. If only testing is preferred, you can set the `TRAIN.ENABLE` to False, and do not forget to pass the path to the model you want to test to `TEST.CHECKPOINT_FILE_PATH`.

```
python tools/run_net.py  --cfg configs/Ego4d/MVIT_B_16x4_CONV.yaml  TRAIN.ENABLE False  TEST.BATCH_SIZE 32  NUM_GPUS 1  OUTPUT_DIR checkpoints/GLC  TEST.CHECKPOINT_FILE_PATH checkpoints/GLC/checkpoints/MViT_Ego4D_ckpt.pyth DATA.PATH_PREFIX ""

# or

python tools/annotate_lerobot_dataset.py --repo-id villekuosmanen/spread_snackbar_box_flaps --cfg configs/Ego4d/MVIT_B_16x4_CONV.yaml  TRAIN.ENABLE False  TEST.BATCH_SIZE 32  NUM_GPUS 1  OUTPUT_DIR checkpoints/GLC  TEST.CHECKPOINT_FILE_PATH checkpoints/GLC/checkpoints/MViT_Ego4D_ckpt.pyth DATA.PATH_PREFIX ""

```

You can also set `DATA.PATH_PREFIX` in configs directly.

You may find it's hard to fully reproduce the results if you train the model again, even though the seed is already fixed. We also observed this issue but failed to fix it. It may be an internal bug in the slowfast codebase, which we build our own model on. However, the difference should be small, and you are still able to get the same number as reported in the paper by running inference with our released weights.
