import torch
import torch.nn as nn
from model import Net  # Replace with your model's file and class name

def check_model():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameter Count: {total_params}")

    # Check for Batch Normalization
    has_batch_norm = any(isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d)
                         for layer in model.modules())
    print(f"Use of Batch Normalization: {'Yes' if has_batch_norm else 'No'}")

    # Check for DropOut
    has_dropout = any(isinstance(layer, nn.Dropout) or isinstance(layer, nn.Dropout2d)
                      for layer in model.modules())
    print(f"Use of DropOut: {'Yes' if has_dropout else 'No'}")

    # Check for Fully Connected Layer or GAP
    has_fc_or_gap = any(isinstance(layer, nn.Linear) or isinstance(layer, nn.AvgPool2d)
                        for layer in model.modules())
    print(f"Use of Fully Connected Layer or GAP: {'Yes' if has_fc_or_gap else 'No'}")

if __name__ == "__main__":
    check_model()
