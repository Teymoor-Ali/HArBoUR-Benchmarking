import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np

class TimeMeasureWrapper(nn.Module):
    def __init__(self, model):
        super(TimeMeasureWrapper, self).__init__()
        self.model = model
        self.layer_times = {}

    def forward(self, x):
        hooks = []

        def hook_fn(module, input, output, layer_name):
            start_time = time.time()
            result = module(input)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if layer_name not in self.layer_times:
                self.layer_times[layer_name] = []
            self.layer_times[layer_name].append(elapsed_time)
            return result

        for name, layer in self.model.named_modules():
            hooks.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name)))

        start_time = time.time()
        x = self.model(x)
        end_time = time.time()
        total_time = end_time - start_time

        for hook in hooks:
            hook.remove()

        return x, total_time, self.layer_times

def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    return image

def main():
    device = torch.device('cpu')
    image_path = "input_image.jpg"
    image = load_image(image_path, device)

    mobilenetv2 = models.mobilenet_v2(pretrained=True).to(device).eval()
    model = TimeMeasureWrapper(mobilenetv2)

    with torch.no_grad():
        output, total_time, layer_times = model(image)

    average_layer_times = {layer: np.mean(times) for layer, times in layer_times.items()}
    predicted_class = output.argmax(dim=1).item()

    print(f"Total Inference Time: {total_time:.6f} seconds")
    print("Average Time per Layer:")
    for layer, avg_time in average_layer_times.items():
        print(f"{layer}: {avg_time:.6f} seconds")
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
