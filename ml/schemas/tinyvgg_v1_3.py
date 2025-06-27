import torch
import torch.nn as nn

class TinyVGG_V1_3(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        # Feature extraction
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units[0],
                  kernel_size=3,
                  stride=1,
                  padding=1),
        # Activation function
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units[0],
                  out_channels=hidden_units[1],
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Dropout2d(p=0.5),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(out_features=128),
      nn.ReLU(),
      nn.Linear(in_features=128, out_features=64),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(in_features=64, out_features=output_shape)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.conv_block_1(x))