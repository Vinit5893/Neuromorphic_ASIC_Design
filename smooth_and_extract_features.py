import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

# Load the EKG dataset (replace 'ekg_data.csv' with your dataset file)
ekg_data = pd.read_csv('100.csv')

# Label the R waves (the big peaks) with SciPy
r_peaks, _ = find_peaks(ekg_data['MLII'], height=1150)

# Smooth out the noisy data
def smooth_data(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

ekg_data['smooth_amplitude'] = smooth_data(ekg_data['MLII'], window_size=5)
ekg_data[0:1000].plot()
plt.grid(True)
plt.show()


# Label the P and T waves using the R waves as reference points
p_start = r_peaks - 50  # Adjust the window size as needed
p_end = r_peaks
t_start = r_peaks
t_end = r_peaks + 50  # Adjust the window size as needed

# Label the Q and S waves
q_start = r_peaks - 10  # Adjust the window size as needed
s_end = r_peaks + 10    # Adjust the window size as needed

# Label multiple heartbeats at once using code from Steps 4–7
heartbeats = pd.DataFrame({
    'R_Peak': r_peaks,
    'P_Start': p_start,
    'P_End': p_end,
    'T_Start': t_start,
    'T_End': t_end,
    'Q_Start': q_start,
    'S_End': s_end
})

# Display the labeled heartbeats
print(heartbeats.head(20))