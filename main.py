import torch
import numpy as np
import os
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


def show_pictures(loader, nb_imgs=0, model=None, batch_size=20):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    if model is not None:
        output = model(images)
        output = output.view(batch_size, 1, 64, 64)
        output = output.detach().numpy()
        fig, axes = plt.subplots(nrows=2, ncols=nb_imgs, sharex=True,
                                 sharey=True, figsize=(25, 4))

        for imgs, row in zip([images, output], axes):
            for img, ax in zip(imgs, row):
                img = np.squeeze(img)
                ax.imshow(img, cmap='gray')
    else:
        images = images.numpy()  # convert images to numpy for display
        fig, axes = plt.subplots(nrows=1, ncols=nb_imgs, sharex=True,
                                 sharey=True, figsize=(25, 4))
        for img, ax in zip(images, axes):
            img = np.squeeze(img)
            ax.imshow(img, cmap='gray')
    plt.show()


def prepare_data(data_root_path, batch_size):

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    transformtest = transforms.Compose([  
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # load the training and test datasets
    train_data = datasets.ImageFolder(root=os.path.join(data_root_path,
                                      'data_train'), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(data_root_path,
                                     'data_test'), transform=transformtest)

    # Create training and test dataloaders
    num_workers = 0

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    return train_loader, test_loader

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)
        
        #self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        #x = self.dropout(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #x = self.dropout(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        #x = self.dropout(x)
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        #x = self.dropout(x)
        x = F.relu(self.t_conv2(x))
        #x = self.dropout(x)
        x = F.relu(self.t_conv3(x))
        #x = self.dropout(x)
        # transpose again, output should have a sigmoid applied
        x = torch.sigmoid(self.conv_out(x))
                
        return x

def train(model, train_loader, n_epochs = 50, noise_factor=0.5, device="cpu", save_path = None):

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        min_train_loss = np.inf

        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
        
            ## add random noise to the input images
            # noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            # noisy_imgs = np.clip(noisy_imgs, 0., 1.)
            # noisy_imgs = noisy_imgs.to(device)
                
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            ## forward pass: compute predicted outputs by passing *noisy* images to the model
            #outputs = model(noisy_imgs)
            images = images.to(device)
            outputs = model(images)
            # calculate the loss
            # the "target" is still the original, not-noisy images        
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
            
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        save_model = ""
        if(train_loss < min_train_loss and save_path is not None):
            save_model = " (model saved)"
            torch.save(model.state_dict(), save_path)
            min_train_loss = train_loss

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            )+save_model)



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
    batch_size = 20

    data_root_path = "/home/danieauf/Data/iris"
    train_loader, test_loader = prepare_data(data_root_path, batch_size)

    #show_pictures(train_loader, 10)

    model = ConvDenoiser()
    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_path = "trained_model.pt"

    #if is_cuda:
    #    model.cuda()
    #train(model, train_loader, device=dev, save_path=model_path)

    model.load_state_dict(torch.load(model_path))

    if(is_cuda):
        model.cpu()

    show_pictures(test_loader, 10, model, batch_size)
    