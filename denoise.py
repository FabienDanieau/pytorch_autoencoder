from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torch
import numpy as np
from ConvDenoiser import ConvDenoiser


def show_img(source, new):
    source = source[0].detach().numpy()
    source = np.transpose(source, (1, 2, 0))
    new = new[0].detach().numpy()
    new = np.transpose(new, (1, 2, 0))

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True,
                             sharey=True, figsize=(25, 4))
    axes[0].imshow(source)
    axes[1].imshow(new)
    plt.show()


if __name__ == '__main__':
    print("denoise a single image")

    img_path = "test.png"
    model_path = "trained_model.pt"

    image = Image.open(img_path).convert('RGB')
    in_transform = transforms.Compose([
                        transforms.Resize(ConvDenoiser.INPUT_SIZE),
                        transforms.ToTensor()
                        ])

    image = in_transform(image)
    image = image.unsqueeze(0)

    model = ConvDenoiser()
    model.load_state_dict(torch.load(model_path))

    is_cuda = torch.cuda.is_available()

    model.eval()
    output = model(image)

    show_img(image, output)
