import torch as t
from datagen import generate_new_device
import matplotlib.pyplot as plt
from model import OutputPredictor

if __name__ == "__main__":
    model = OutputPredictor()
    params = t.load("fdtd.pth")
    model.load_state_dict(params)

    epsr, input, output = generate_new_device()
    epsr = t.tensor(epsr).float()
    input = t.tensor(input)
    input_real = input.real.clone().detach().float()
    input_imag = input.imag.clone().detach().float()

    print(output)

    with t.no_grad():
        output_pred = model(
            epsr.unsqueeze(0), input_real.unsqueeze(0), input_imag.unsqueeze(0)
        )[0]

    print(output_pred.shape)
    output_pred_real = output_pred[:20]
    print(output_pred_real)
    output_pred_imag = output_pred[20:]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(output_pred_real, label="Predicted Real")
    plt.plot(output.real, label="True Real")
    plt.legend()
    plt.title("Predicted Output")
    plt.show()
