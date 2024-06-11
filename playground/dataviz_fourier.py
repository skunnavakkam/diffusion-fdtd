import matplotlib.pyplot as plt
import pickle
from numpy.fft import fft
import numpy as np

if __name__ == "__main__":
    # loading data
    with open("../data_raw.pkl", "rb") as f:
        epsr_arr, input_arr, output_arr = pickle.load(f)

    while True:
        # generate a random index
        idx = np.random.randint(0, len(epsr_arr))
        epsr = epsr_arr[idx]
        input = input_arr[idx]
        output = output_arr[idx]

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(fft(input).real, label="Input Real")
        print(fft(input))
        plt.subplot(2, 2, 2)
        plt.plot(fft(input).imag, label="Input Imag")
        plt.subplot(2, 2, 3)
        plt.plot(fft(output).real, label="Output Real")
        plt.subplot(2, 2, 4)
        plt.plot(fft(output).imag, label="Output Imag")
        plt.legend()
        plt.title("Input and Output")
        plt.show()
