
from monai.data.meta_tensor import MetaTensor
from omegaconf import OmegaConf

from collections import OrderedDict
import torch
import torch.nn as nn
from src.models.lightning_module import SegmentationModule
from src.utils import *
import os
import glob
from pathlib import Path

#export PYTHONPATH=$PYTHONPATH:$(pwd) && python3 src/utils/soup_model.py

torch.serialization.add_safe_globals([MetaTensor])
torch.set_float32_matmul_precision('medium')

def model_soup(models, method="uniform"):
    """
    Perform model soup by averaging weights of multiple models.
    
    Args:
        models (list): List of PyTorch models (must have identical architectures).
        method (str): "uniform" (default) or "greedy" for weight averaging strategy.
        
    Returns:
        torch.nn.Module: A new model with averaged weights.
    """
    assert len(models) > 0, "Provide at least one model for souping."
    
    # Get state_dicts from all models
    # Check if models are wrapped in SegmentationModule or are raw models
    state_dicts = []
    for model in models:
        if isinstance(model, SegmentationModule):
   
             if model.ema is not None:
                 print("Using EMA weights for souping from a model with EMA.")
                 state_dicts.append(model.ema.module.state_dict())
             else:
                 print("Using standard model weights for souping.")
                 state_dicts.append(model.model.state_dict())
        elif isinstance(model, nn.Module):
             state_dicts.append(model.state_dict())
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    
    # Create an empty OrderedDict for the new model's weights
    avg_state_dict = OrderedDict()

    # Uniform Model Soup (Simple Weight Averaging)
    if method == "uniform":
        print(f"Averaging {len(state_dicts)} models...")
        for key in state_dicts[0]:  # Iterate over parameter keys
            weights = [sd[key] for sd in state_dicts]
            
            avg_state_dict[key] = torch.mean(torch.stack(weights), dim=0)

                 
    else:
        raise ValueError(f"Unknown method: {method}. Supported: ['uniform']")
    
    if isinstance(models[0], SegmentationModule):
        if models[0].ema is not None:
             models[0].ema.module.load_state_dict(avg_state_dict)
             models[0].model.load_state_dict(avg_state_dict)
        else:
             models[0].model.load_state_dict(avg_state_dict)
        return models[0]
    else:
        models[0].load_state_dict(avg_state_dict)
        return models[0]


if __name__ == "__main__":
    
    # Hardcoded checkpoints as requested
    CHECKPOINT_PATHS = [
        #"resUnet-ema-fold0/best-epoch=449-val_loss=0.3746-val_dice=0.5755.ckpt",
        #"resUnet-ema-fold0/best-epoch=439-val_loss=0.3751-val_dice=0.5753.ckpt"
        "cv_outputs_res_unet_new_dataset_4loss/fold_0/fold0_4loss_new_dataset/best-epoch=369-val_loss=0.3769-val_dice=0.5727.ckpt",
        "cv_outputs_res_unet_new_dataset_4loss/fold_0/fold0_4loss_new_dataset/best-epoch=379-val_loss=0.3783-val_dice=0.5711.ckpt"
    ]
    
    print(f"Loading {len(CHECKPOINT_PATHS)} models for souping...")
    models = []
    
    # Keep the first raw checkpoint to reuse its metadata structure
    first_checkpoint_payload = None
    
    for i, ckpt_path in enumerate(CHECKPOINT_PATHS):
        print(f"Loading: {ckpt_path}")
        try:
            # We load the full checkpoint manually to preserve metadata (like pytorch-lightning_version)
            # if we want to save a valid PL checkpoint later.
            if i == 0:
                first_checkpoint_payload = torch.load(ckpt_path, map_location='cpu')
            
            model = SegmentationModule.load_from_checkpoint(ckpt_path)
            model.eval() # Set to eval mode
            models.append(model)
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
            raise e

    print("Creating Uniform Model Soup...")
    soup_module = model_soup(models)
    
    output_path = "souped_res_unet.pth" 
    
    print(f"Saving souped model to {output_path}...")
    
    # Saving as a valid PyTorch Lightning checkpoint
    # We use the structure of the first checkpoint but replace the state_dict
    if first_checkpoint_payload is not None:
        # Update the state_dict in the payload
        # Note: SegmentationModule state_dict includes "model." prefix and potentially "ema." prefix
        # Our soup_module IS a SegmentationModule with updated weights.
        first_checkpoint_payload['state_dict'] = soup_module.state_dict()
        
   
        
        torch.save(first_checkpoint_payload, output_path)
    else:
        # Fallback (shouldn't happen if loop runs)
        torch.save(soup_module.state_dict(), output_path)
    
    # Also save just the model weights (standard pytorch format) for easier pure-pytorch loading
    torch.save(soup_module.model.state_dict(), "souped_res_unet_weights.pth")
    print("Done.")
