import torch
import numpy as np
import os
import glob
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image, ImageStat


def reverse_normalization(batch, mean, std):
    x = batch.new(*batch.size())
    x[:, 0, :, :] = batch[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = batch[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = batch[:, 2, :, :] * std[2] + mean[2]
    return x


def show_pictures(loader, nb_imgs=0, model=None, batch_size=20,
                  mean=None, std=None):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    if model is not None:
        output = model(images)
        #output = output.view(batch_size, 3, 64, 64)
        output = reverse_normalization(output, mean, std)
        output = output.detach().numpy()

        images = reverse_normalization(images, mean, std)
        images = images.numpy()

        fig, axes = plt.subplots(nrows=2, ncols=nb_imgs, sharex=True,
                                 sharey=True, figsize=(25, 4))

        for imgs, row in zip([images, output], axes):
            for img, ax in zip(imgs, row):
                img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)
    else:
        if mean is not None and std is not None:
            images = reverse_normalization(images, mean, std)
        images = images.numpy()  # convert images to numpy for display
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
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transformtest = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
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


class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # encoder layers #
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers #
        # transpose layer, a kernel of 2 and a stride of 2
        # will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 64, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # encode #
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x = self.dropout(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = self.dropout(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        # x = self.dropout(x)

        # decode #
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # x = self.dropout(x)
        x = F.relu(self.t_conv2(x))
        # x = self.dropout(x)
        x = F.relu(self.t_conv3(x))
        # x = self.dropout(x)
        # transpose again, output should have a sigmoid applied
        x = torch.sigmoid(self.conv_out(x))
      
        return x


def train(model, train_loader, valid_loader, n_epochs=50, noise_factor=0.5,
          device="cpu", save_path=None):

    min_valid_loss = np.inf
    losses = {'train': [], 'validation': []}

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data in train_loader:
            images, _ = data
                        
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            images = images.to(device)
            outputs = model(images)
            # calculate the loss
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
            images = images.to(device)
            output = model(images)
            #  calculate the loss
            loss = criterion(output, images)
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
            min_valid_loss = train_loss

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss)
            + ' Valid loss {:.6f}: '.format(valid_loss)
            + save_model)

    return losses


def test(model, test_loader, device="cpu"):
    test_loss = 0.0
    model.eval()  # prep model for evaluation
    for images, _ in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        images = images.to(device)
        output = model(images)
        #  calculate the loss
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

    show_pictures(train_loader, 10, mean=mean_train, std=std_train)

    model = ConvDenoiser()
    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_path = "trained_model.pt"

    # if is_cuda:
    #    model.cuda()
    #losses = train(model, train_loader, valid_loader, n_epochs=50, device=dev, save_path=model_path)

    # display_losses(losses, True)

    model.load_state_dict(torch.load(model_path))

    if is_cuda:
        model.cuda()
    test(model, test_loader, device=dev)

    if(is_cuda):
        model.cpu()

    show_pictures(test_loader, 10, model, batch_size, mean_train, std_train)
