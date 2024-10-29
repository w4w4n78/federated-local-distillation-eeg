import torch
import numpy as np
import random
import argparse

from torcheeg.models import ArjunViT
from torcheeg.trainers import ClassifierTrainer
import pytorch_lightning as pl

from utils import set_seed
from dataset import prepare_dataset, prepare_dataloaders

# Define functions to create new model instances
def create_default_model(dataset_name):
    if dataset_name == 'deap':
        return ArjunViT(
            num_electrodes=32,
            chunk_size=128,
            t_patch_size=128//4,
            depth=6,
            heads=8,
            hid_channels=32,
            head_channels=64,
            mlp_channels=64,
            embed_dropout=0,
            dropout=0,
        )
    elif dataset_name == 'seed':
        return ArjunViT(
            num_electrodes=62,
            chunk_size=200,
            t_patch_size=200//4,
            depth=6,
            heads=8,
            hid_channels=32,
            head_channels=64,
            mlp_channels=64,
            embed_dropout=0,
            dropout=0,
        )
    elif dataset_name == 'dreamer':
        return ArjunViT(
            num_electrodes=14,
            chunk_size=128,
            t_patch_size=128//4,
            depth=6,
            heads=8,
            hid_channels=32,
            head_channels=64,
            mlp_channels=64,
            embed_dropout=0,
            dropout=0,
        )

def create_optimized_model(dataset_name):
    if dataset_name == 'deap':
        return ArjunViT(
            num_electrodes=32,
            chunk_size=128,
            t_patch_size=128//4,
            depth=6,
            heads=8,
            hid_channels=256,
            head_channels=1024,
            mlp_channels=1024,
            embed_dropout=0,
            dropout=0.3,
        )
    elif dataset_name == 'seed':
        return ArjunViT(
            num_electrodes=62,
            chunk_size=200,
            t_patch_size=200//4,
            depth=6,
            heads=8,
            hid_channels=512,
            head_channels=4,
            mlp_channels=4,
            embed_dropout=0.2,
            dropout=0.5,
        )
    elif dataset_name == 'dreamer':
        return ArjunViT(
            num_electrodes=14,
            chunk_size=128,
            t_patch_size=128//4,
            depth=6,
            heads=8,
            hid_channels=256,
            head_channels=128,
            mlp_channels=1024,
            embed_dropout=0.2,
            dropout=0.1,
        )
    
def main(seed_number, dataset, overlap_percent, batch_size, lr, test_ratio, epochs):
    idx = {
        'deap': 0,
        'seed': 1,
        'dreamer': 2
    }[dataset]

    set_seed(seed_number)
            
    datasets = prepare_dataset(feature_type="raw_normalized", class_type="binary", overlap_percent=overlap_percent)
    trainloaders, valloaders, testloaders = prepare_dataloaders(
        datasets, batch_size=batch_size, test_ratio=test_ratio, seed=seed_number
    )
    
    model = create_optimized_model(dataset)
    
    trainer = ClassifierTrainer(model, lr=lr, num_classes=2, accelerator="auto")
    trainer.fit(trainloaders[idx], valloaders[idx],
                max_epochs=epochs, deterministic=True,
                default_root_dir=f'models/client/seed_number_{seed_number}/{dataset}/fit',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)])
    trainer.test(testloaders[idx], deterministic=True, enable_checkpointing=False,
                default_root_dir=f'models/client/seed_number_{seed_number}/{dataset}/test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument(
        "--seed_number", 
        type=int, 
        required=True, 
        help="Seed number (must be an integer)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=['deap', 'seed', 'dreamer'], 
        required=True, 
        help="Dataset (must be 'deap', 'seed', or 'dreamer')"
    )
    parser.add_argument(
        "--overlap_percent", 
        type=int, 
        choices=[0, 25, 50, 75], 
        required=True, 
        help="Overlap percent (must be 0, 25, 50, or 75)"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Batch size (must be an integer, default is 64)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.00001, 
        help="Learning rate (must be numeric, default is 0.00001)"
    )
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.2, 
        help="Test ratio (must be between 0 and 1, default is 0.2)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=300, 
        help="Number of epochs (must be an integer, default is 300)"
    )

    # Parse and validate arguments
    args = parser.parse_args()

    # Additional validation for test_ratio
    if not (0 <= args.test_ratio <= 1):
        parser.error("test_ratio must be between 0 and 1.")

    main(args.seed_number, args.dataset, args.overlap_percent, args.batch_size, args.lr, args.test_ratio, args.epochs)