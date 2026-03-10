# Vesuvius Challenge 10th Place Solution

## Pipeline Steps (Reproducibility)

| Step | Script | Status | Description |
|------|--------|--------|-------------|
| 1 | `step1_download_kaggle_data.sh` | TODO | Download competition data from Kaggle |
| 2 | `step2_convert_to_npy.sh` | TODO | Convert TIF images/labels to NPY format |
| 3 | `step3_download_pretrain_data.sh` | TODO | Download additional labeled data for pretraining |
| 4 | `step4_pretrain.sh` | TODO | Pretrain ResEncL, Primus, PrimusV2 on additional data |
| 5 | `step5_train_5fold.sh` | TODO | Train 5-fold CV for each model (fine-tune from pretrained) |
| 6 | `step6_generate_oof.sh` | TODO | Generate OOF predictions for each model |
| 7 | `step7_train_2nd_stage.sh` | TODO | Train 2nd stage 5-fold (image + 3 OOF channels) |
| 8-? | TBD | TODO | Future steps (generate 2nd stage OOF, 3rd+ stages, deformnet, submission) |

## Key Architecture
- **3 first-stage models**: ResEncL (residual encoder UNet), Primus, PrimusV2
- **Multi-stage pipeline**: 1st stage → 2nd stage refinement → 3rd/4th/5th stage → DeformNet
- **OOF caching**: Each 1st-stage model's OOF predictions become input channels for 2nd stage

## Directory Structure
```
train.csv                    # Competition CSV
train_images/                # Raw TIF images (from Kaggle)
train_labels/                # Raw TIF labels (from Kaggle)
train_images_npy/            # Converted NPY images
train_labels_npy/            # Converted NPY labels
train_skeletons_npy/         # Precomputed skeletons
additional_data/             # Pretrain data (from ash2txt.org)
  images/, labels/, samples.csv
pretrained_checkpoints/      # Pretrained model weights
  resencl/, primus/, primus_v2/
cv_outputs_resencl/          # 5-fold training outputs
cv_outputs_primus/
cv_outputs_primus_v2/
1st_stage_cache/             # ResEncL OOF probabilities
Primus_1st_stage_cache/      # Primus OOF probabilities
PrimusV2_1st_stage_cache/    # PrimusV2 OOF probabilities
cv_outputs_2nd_stage/        # 2nd stage 5-fold outputs
```

## Model Configs (from train_cv.py defaults + eval script)
- **ResEncL**: model_type=unet, channels=[32,64,128,256,320,320], strides=[1,2,2,2,2,2], blocks=[1,3,4,6,6,6]
- **Primus**: model_type=primus, don't use drop_path_rate
- **PrimusV2**: model_type=primus_v2, uses drop_path_rate 0.2
- All use: deep_supervision, EMA, CutMix, loss_weights=[0.4, 0.4, 0.1, 0.1]
- Patch size: 160 for training, 160 for inference
