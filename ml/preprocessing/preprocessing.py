from torchvision.transforms import v2
import torch

eval_transform = v2.Compose([
    # for grayscale
    v2.Grayscale(num_output_channels=1),
    # resize image
    v2.Resize(size=(64, 64)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5]), # very important, increase model performance by a lot beside tweaking lr on optimizer
])