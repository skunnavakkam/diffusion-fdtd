import numpy as np
from datagen import generate_new_device
import torch.nn as nn
import torch.optim as optim
import torch as t
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import os
from model import OutputPredictor
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # check if the data exists
    if not os.path.exists("data_raw.pkl"):
        # generating data first
        epsr_arr = []
        input_arr = []
        output_arr = []
        for i in range(100000):
            epsr, input, output = generate_new_device()
            epsr_arr.append(epsr)
            input_arr.append(input)
            output_arr.append(output)
            if i % 10 == 0:
                print(f"Generated {i} devices")

        # pickle
        with open("data_raw.pkl", "wb") as f:
            pickle.dump((epsr_arr, input_arr, output_arr), f)
    else:
        with open("data_raw.pkl", "rb") as f:
            epsr_arr, input_arr, output_arr = pickle.load(f)

    epsrs = t.tensor(np.array(epsr_arr)).float()
    inputs = t.tensor(np.array(input_arr))
    inputs_real = inputs.real.clone().detach().float()
    inputs_imag = inputs.imag.clone().detach().float()
    outputs = t.tensor(np.array(output_arr))
    temp = []
    for output in outputs:
        new = t.cat((output.real, output.imag))
        temp.append(new)
    outputs = t.stack(temp).float()

    # split the data into training and testing
    dataset = TensorDataset(epsrs, inputs_real, inputs_imag, outputs)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # initialize the model
    model = OutputPredictor().float()
    model.float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # hyperparams
    num_epochs = 4

    # training loop
    loss_arr = []
    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            epsr, input_real, input_imag, output = data
            optimizer.zero_grad()

            # forward
            outputs = model(epsr, input_real, input_imag)
            loss = criterion(outputs, output)

            # backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss}")
                loss_arr.append(running_loss)
                running_loss = 0

    plt.plot(loss_arr)
    plt.autoscale()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # test on the test set
    with t.no_grad():
        for i, data in enumerate(test_loader, 0):
            epsr, input_real, input_imag, output = data
            outputs = model(epsr, input_real, input_imag)
            loss = criterion(outputs, output)
            print(f"Test loss: {loss}")

    t.save(model.state_dict(), "fdtd.pth")
