# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

# Dataset loading
my_transform = torchvision.transforms.ToTensor()
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform= my_transform)
val_dataset = dsets.MNIST(root='./data', train=False, download=True, transform= my_transform)

# Paths
# models_path = "..\jacanram\Desktop\Doctorado\Formación\Curso Pytorch\MNIST problem\models"
models_path = "../models"
print(models_path)
# Data visualization
def show_data(data_sample):
    image = data_sample[0].numpy()
    print("Input image size: ", np.shape(image))
    plt.imshow(image.reshape(np.shape(image)[-2], np.shape(image)[-1]), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))
    plt.show()

show_data(train_dataset[3])


# Convolutional model constructor
class CNN(nn.Module):

    # Contructor
    def __init__(self, Layers, Kernels, MaxPoolKernels, Stride, Padding, BatchNorm=True):
        super(CNN, self).__init__()
        self.hidden = nn.ModuleList()
        self.maxpool = nn.ModuleList()
        self.bn = nn.ModuleList()
        idx = 0
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=Kernels[idx],
                                         stride=Stride[idx], padding=Padding[idx]))
            if BatchNorm == True:
                self.bn.append(nn.BatchNorm2d(output_size))
            if MaxPoolKernels[idx] > 0:
                self.maxpool.append(nn.MaxPool2d(kernel_size=MaxPoolKernels[idx]))
                img_size = int(img_size / MaxPoolKernels[idx])
                idx += 1

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, cnn) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.relu(cnn(activation))
                if self.bn[l]:
                    activation = self.bn[l](activation)
                if MaxPoolKernels[l] > 0:
                    activation = self.maxpool[l](activation)
            else:
                activation = torch.relu(cnn(activation))
                if self.bn[l]:
                    activation = self.bn[l](activation)
                if MaxPoolKernels[l] > 0:
                    activation = self.maxpool[l](activation)
                activation = activation.view(activation.size(0), -1)

        return activation

    # Outputs in each steps
    def activations(self, x):
        L = len(self.hidden)
        activations = []
        for (l, cnn) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.relu(cnn(x))
                activation = self.maxpool[l](activation)
            else:
                activation = torch.relu(cnn(activation))
                activation = self.maxpool[l](activation)
                activation = activation.view(activation.size(0), -1)
            activations.append(activation)
        return activations

# Fully connected model constructor
class Net(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2

# CNN hyperparameters definition
Layers= [1, 16, 16]
Kernels= [3, 3]
MaxPoolKernels= [0, 0]
Padding = [0, 0]
Stride = [1, 1]

# We create and check the parameters of the model
CNN_model = CNN(Layers, Kernels, MaxPoolKernels, Stride, Padding)
# CNN_model.state_dict()

# We calculate the output size of the CNN (MaxPool with no overlapping)
output_size = 28 # Input image size
idx = 0
while idx < len(Layers)-1:
  output_size = (output_size + 2*Padding[idx] - Kernels[idx])/Stride[idx] + 1
  if MaxPoolKernels[idx] > 0:
    output_size = (output_size/MaxPoolKernels[idx])
  idx = idx+1
output_size = np.floor(output_size)
print("The output size after the convolutional layers is: ", output_size)

# Fully connected layer hyperparameters
input_size = int(Layers[-1]*(output_size)**2)
print("The size after flattening is: ", input_size)
Linear_Layers = [input_size, 512, 10]

# We create the linear model
linear_model = Net(Linear_Layers)

# We assemble the models
model = MyEnsemble(CNN_model, linear_model)

# Train the model

# Define hyperparameters
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=5000)

n_epochs = 1
cost_list = []
accuracy_list = []
N_test = len(val_dataset)
COST = 0


def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += loss.data

        cost_list.append(COST)
        correct = 0
        # perform a prediction on the validation  data
        for x_test, y_test in validation_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)


train_model(n_epochs)

torch.save(model.state_dict(), models_path)
# model = loaded_model(*args, **kwargs)
# model.load_state_dict(torch.load(models_path))

# Plot the loss and accuracy

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.set_xlabel('epoch', color=color)
ax2.plot(accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()
plt.show()

# We compare the predictions with the labels
# Plot the mis-classified samples

count = 0
for x, y in torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1):
    z = model(x)
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("yhat: ",yhat)
        count += 1
    if count >= 5:
        break

# Define the function for plotting the activations

def plot_activations(A, number_rows=1, name="", i=0):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()
    fig, axes = plt.subplots(number_rows, n_activations // number_rows)
    fig.subplots_adjust(hspace = 0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Set the label for the sub-plot.
            ax.set_xlabel("activation:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

activations = CNN_model.activations(val_dataset)
plot_activations(activations[0],number_rows=1,name=" feature map")