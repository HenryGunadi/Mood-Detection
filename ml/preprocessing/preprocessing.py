from torchvision.transforms import v2
import torch

eval_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize(size=(64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5]),
])