import torch as t
from datagen import generate_new_device
import matplotlib.pyplot as plt
from model import OutputPredictor
from main import FOURIER
from numpy.fft import fft, ifft

if __name__ == "__main__":
    model = OutputPredictor()
    params = t.load("fdtd.pth")
    model.load_state_dict(params)

    counter = 0
    while True:
        epsr, input, output = generate_new_device()
        epsr = t.tensor(epsr).float()
        input = t.tensor(input)
        input_real = input.real.clone().detach().float()
        input_imag = input.imag.clone().detach().float()
        output *= 1e8
        # print(output)

        model.eval()
        with t.no_grad():
            output_pred = model(
                epsr.unsqueeze(0), input_real.unsqueeze(0), input_imag.unsqueeze(0)
            )[0]

        # print(output_pred.shape)
        # print(output_pred_real)
        output_pred_real = output_pred[:20]
        output_pred_imag = output_pred[20:]

        if FOURIER:
            new_output = output_pred_real + 1j * output_pred_imag
            new_output = ifft(new_output) * 10
            output_pred_real = new_output.real
            output_pred_imag = new_output.imag

        print(output_pred)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(output_pred_real, label="Predicted Real")
        plt.plot(output.real, label="True Real")
        # add a new subplot for imag
        plt.subplot(1, 2, 2)
        plt.plot(output_pred_imag, label="Predicted Imag")
        plt.plot(output.imag, label="True Imag")
        plt.legend()
        plt.title("Predicted Output")
        plt.show()
        print(counter)
        counter += 1
