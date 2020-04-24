import torch
import numpy as np
import os
import glob
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image, ImageStat, ImageDraw
from ConvDenoiser import ConvDenoiser


def reverse_normalization(batch, mean, std):
    x = batch.new(*batch.size())
    x[:, 0, :, :] = batch[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = batch[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = batch[:, 2, :, :] * std[2] + mean[2]
    return x


def add_noise_to_images(list_im):
    for im in list_im:
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray((im * 255).astype(np.uint8))
        draw = ImageDraw.Draw(im)
        draw.line((0, 0) + im.size, fill=128)
        draw.line((0, im.size[1], im.size[0], 0), fill=128)


def show_pictures(loader, nb_imgs=0, model=None, batch_size=20,
                  mean=None, std=None):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    if model is not None:
        transform = transforms.Compose([transforms.RandomErasing(p=1)])
        noisy_images = batch_transform(images, transform)
        output = model(noisy_images)

        # output = reverse_normalization(output, mean, std)
        output = output.detach().numpy()

        # images = reverse_normalization(images, mean, std)
        noisy_images = noisy_images.detach().numpy()

        fig, axes = plt.subplots(nrows=2, ncols=nb_imgs, sharex=True,
                                 sharey=True, figsize=(25, 4))

        for imgs, row in zip([noisy_images, output], axes):
            for img, ax in zip(imgs, row):
                img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)
    else:
        # if mean is not None and std is not None:
        #    images = reverse_normalization(images, mean, std)
        images = images.detach().numpy()  # convert images to numpy for display
        fig, axes = plt.subplots(nrows=1, ncols=nb_imgs, sharex=True,
                                 sharey=True, figsize=(25, 4))
        for img, ax in zip(images, axes):
            img = np.transpose(img, (1, 2, 0))
            ax.imshow(img)
    plt.show()


def display_losses(losses, save=False):
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    _ = plt.ylim()
    if not save:
        plt.show()
    else:
        print("save losses to losses.png")
        plt.savefig('losses.png')
        plt.close()


def get_mean_and_std(path):
    print("computing mean and stddev of "+path)
    image_files = [f for f in glob.glob(os.path.join(path, '*.png'))]
    print(str(len(image_files)) + " image(s) found.")

    mean = np.zeros(3)
    stddev = np.zeros(3)

    for filepath in image_files:
        img = Image.open(filepath)
        mean += ImageStat.Stat(img).mean
        stddev += ImageStat.Stat(img).stddev

    mean /= len(image_files)
    stddev /= len(image_files)
    mean /= 255.0
    stddev /= 255.0

    return mean, stddev


def split_indices_train(train_data, valid_size):
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


def prepare_data(data_root_path, batch_size, valid_size,
                 mean, std):

    transform = transforms.Compose([
        transforms.Resize(ConvDenoiser.INPUT_SIZE),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    transformtest = transforms.Compose([
        transforms.Resize(ConvDenoiser.INPUT_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    # load the training and test datasets
    train_data = datasets.ImageFolder(root=os.path.join(data_root_path,
                                      'data_train'), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(data_root_path,
                                     'data_test'), transform=transformtest)

    # Create training and test dataloaders
    num_workers = 0

    train_idx, valid_idx = split_indices_train(test_data, valid_size)
    test_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def batch_transform(batch, transform):
    new_batch = batch.new(*batch.size())
    for i in range(len(batch)):
        new_batch[i] = transform(batch[i])
    return new_batch


def train(model, train_loader, valid_loader, n_epochs=50, noise_factor=0.5,
          device="cpu", save_path=None):

    min_valid_loss = np.inf
    losses = {'train': [], 'validation': []}

    # add noise to image
    transform = transforms.Compose([
        transforms.RandomErasing()
    ])

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data in train_loader:
            images, _ = data

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            noisy_images = batch_transform(images, transform)
            noisy_images = noisy_images.to(device)
            outputs = model(noisy_images)
            # calculate the loss
            images = images.to(device)
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to
            #  model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)

        model.eval()  # prep model for evaluation
        for images, _ in valid_loader:
            # forward pass: compute predicted outputs by passing
            # inputs to the model
            noisy_images = batch_transform(images, transform)
            noisy_images = noisy_images.to(device)
            outputs = model(noisy_images)

            images = images.to(device)
            #  calculate the loss
            loss = criterion(outputs, images)
            # update running validation loss
            valid_loss += loss.item()*images.size(0)

        # print avg training statistics
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)
        save_model = ""
        if(valid_loss < min_valid_loss and save_path is not None):
            save_model = " (model saved)"
            torch.save(model.state_dict(), save_path)
            min_valid_loss = valid_loss

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss)
            + ' Valid loss {:.6f}: '.format(valid_loss)
            + save_model)

    return losses


def test(model, test_loader, device="cpu"):
    test_loss = 0.0

    # add noise to image
    transform = transforms.Compose([
        transforms.RandomErasing()
    ])

    model.eval()  # prep model for evaluation
    for images, _ in test_loader:
        # forward pass: compute predicted outputs by passing inputs
        #  to the model
        noisy_images = batch_transform(images, transform)
        noisy_images = noisy_images.to(device)
        output = model(noisy_images)
        #  calculate the loss
        images = images.to(device)
        loss = criterion(output, images)
        # update running test loss
        test_loss += loss.item()*images.size(0)

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))


if __name__ == '__main__':
    print("test pytorch encoder/decoder")

    # check if CUDA is available
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        dev = "cuda"
        print("cuda is available")
    else:
        dev = "cpu"
        print("cuda is not available")

    # how many samples per batch to load
    batch_size = 10

    data_root_path = "/home/danieauf/Data/iris"

    # m, s = get_mean_and_std(os.path.join(data_root_path, 'data_test/test'))
    # print(m, s)
    mean_test = (0.38698485, 0.28999359, 0.20289349)
    std_test = (0.31539622, 0.26247943, 0.20613219)

    # m, s = get_mean_and_std(os.path.join(data_root_path, 'data_train/train'))
    # print(m, s)
    mean_train = (0.36651049, 0.24841637, 0.16446467)
    std_train = (0.30748653, 0.24475968, 0.1933519)

    train_loader, valid_loader, test_loader = prepare_data(data_root_path,
                                                           batch_size,
                                                           0.4,
                                                           mean_train,
                                                           std_train)

    #show_pictures(train_loader, 10, mean=mean_train, std=std_train)

    model = ConvDenoiser()
    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_path = "trained_model.pt"

    if is_cuda:
        model.cuda()
    losses = train(model, train_loader, valid_loader, n_epochs=20, device=dev,
                   save_path=model_path)

    display_losses(losses, True)

    model.load_state_dict(torch.load(model_path))

    if is_cuda:
        model.cuda()
    test(model, test_loader, device=dev)

    if(is_cuda):
        model.cpu()

    show_pictures(test_loader, 10, model, batch_size, mean_train, std_train)
