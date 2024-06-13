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
from tqdm import tqdm
from numpy.fft import fft


FOURIER = True
hyperparams = {
    "num_epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.01,
    "weight_decay": 0.001,
    "patience": 5,
    "factor": 0.5,
}


device = None
if t.backends.mps.is_available():
    device = t.device("mps")
    print("Using MPS")
elif t.cuda.is_available():
    device = t.device("cuda")
    print("Using CUDA")
else:
    device = t.device("cpu")
    print("Using CPU")


if __name__ == "__main__":
    if not os.path.exists("data_raw.pkl"):
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

        with open("data_raw.pkl", "wb") as f:
            pickle.dump((epsr_arr, input_arr, output_arr), f)
    else:
        with open("data_raw.pkl", "rb") as f:
            epsr_arr, input_arr, output_arr = pickle.load(f)

    if FOURIER:
        epsrs = t.tensor(np.array(epsr_arr)).float().to(device)
        inputs = np.array(input_arr)
        inputs = t.tensor(fft(inputs, axis=1))
        inputs_real = inputs.real.clone().detach().float().to(device)
        inputs_imag = inputs.imag.clone().detach().float().to(device)

        outputs = np.array(output_arr)
        outputs = t.tensor(fft(outputs, axis=1))
        outputs = (t.cat((outputs.real, outputs.imag), dim=1) * 1e7).float().to(device)
    else:
        epsrs = t.tensor(np.array(epsr_arr)).float().to(device)
        inputs = t.tensor(np.array(input_arr))
        inputs_real = inputs.real.clone().detach().float().to(device)
        inputs_imag = inputs.imag.clone().detach().float().to(device)

        outputs = t.tensor(np.array(output_arr))
        outputs = (t.cat((outputs.real, outputs.imag), dim=1) * 1e8).float().to(device)

    # split the data into training and testing
    dataset = TensorDataset(epsrs, inputs_real, inputs_imag, outputs)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    model = OutputPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    num_epochs = hyperparams["num_epochs"]
    loss_arr = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"
        ) as pbar:
            for epsr, input_real, input_imag, output in train_loader:
                optimizer.zero_grad()
                predictions = model(epsr, input_real, input_imag)
                loss = criterion(predictions, output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
        loss_arr.append(epoch_loss / len(train_loader))
        scheduler.step(epoch_loss)

    plt.plot(loss_arr)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.show()

    with t.no_grad():
        total_loss = 0.0
        for epsr, input_real, input_imag, output in tqdm(test_loader, desc="Testing"):
            predictions = model(epsr, input_real, input_imag)
            loss = criterion(predictions, output)
            total_loss += loss.item()
        print(f"Average Test Loss: {total_loss / len(test_loader):.4f}")

    t.save(model.state_dict(), "fdtd.pth")
