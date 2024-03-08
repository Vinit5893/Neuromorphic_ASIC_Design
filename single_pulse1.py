import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from biosppy.signals import ecg


file_number = 100

def path_to_csv(file_number):
    filenumber = str(file_number) + '.csv'
    for dirname, _, filenames in os.walk('mitbih_database/'):
        for filename in filenames:
            if (filename == filenumber):
                file_path = os.path.join(dirname, filename)
                # print(os.path.join(dirname, file_number))
                return file_path

csv_path = path_to_csv(file_number)
# print(csv_path)

def path_to_txt(file_number):
    filenumber = str(file_number) + 'annotations.txt'
    for dirname, _, filenames in os.walk('mitbih_database/'):
        for filename in filenames:
            if (filename == filenumber):
                file_path = os.path.join(dirname, filename)
                # print(os.path.join(dirname, file_number))
                return file_path

txt_path = path_to_txt(file_number)
# print(txt_path)

DataFrame = pd.read_csv(csv_path, delimiter=',')
DataFrame.rename(columns = lambda x: x.replace("'","").replace(" ","_"), inplace=True)
# print(DataFrame.head(20))

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
# Visualizing the ECG Signal

def plot_Signals(file_number):
    path = path_to_csv(file_number)
    df = pd.read_csv(path, delimiter=',')
    df.rename(columns = lambda x: x.replace("'","").replace(" ","_"), inplace=True)

    # Extract the data from the DataFrame
    samples = df[df.columns[0]]  
    signal_1 = df[df.columns[1]]  
    signal_2 = df[df.columns[2]]  

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    # Tracer le signal 1
    axes[0].plot(samples, signal_1, label="Signal 1", color='blue')
    axes[0].set_xlabel("Frame ( 1/360 secondes)")
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(df.columns[1])
    axes[0].legend()

    # Tracer le signal 2
    axes[1].plot(samples, signal_2, label="Signal 2", color='red')
    axes[1].set_xlabel("Frame ( 1/360 secondes)")
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(df.columns[2])
    axes[1].legend()

    # Afficher le tracé
    plt.tight_layout()
    plt.show()
    
    signal_1_stats = df[df.columns[1]].describe()
    print('Statistiques descriptives pour Signal 1 :\n', signal_1_stats)
    print()
    print()
    signal_2_stats = df[df.columns[2]].describe()
    print('Statistiques descriptives pour Signal 2 :\n', signal_2_stats)

# plot_Signals(104)

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
# Visualizing the peaks of r wave of ECG Signal

def detect_and_visualize_r_peaks(file_number):
    path = path_to_csv(file_number)
    df = pd.read_csv(path, delimiter=',')
    df.rename(columns = lambda x: x.replace("'","").replace(" ","_"), inplace=True)

    # Extract the data from the DataFrame
    samples = df[df.columns[0]]  
    signal_1 = df[df.columns[1]]  
    signal_2 = df[df.columns[2]]  

    # Label the R waves (the big peaks) with SciPy in MLII
    # r_peaks_mlii, _ = find_peaks(signal_1, height=0.6, distance=200)
    r_peaks_mlii = ecg.ssf_segmenter(signal=signal_1, sampling_rate=1000.0, threshold=150, before=0.2, after=0.2)
    r_peaks_mlii = r_peaks_mlii['rpeaks']
    # print(r_peaks_mlii)

    # Label the R waves (the big peaks) with SciPy in V5
    r_peaks_v5 = ecg.ssf_segmenter(signal=signal_2, sampling_rate=1000.0, threshold=150, before=0.1, after=0.1)
    r_peaks_v5 = r_peaks_v5['rpeaks']

    # visualisation of the r peaks in MLII
    plt.figure(figsize=(14, 6))
    plt.plot(signal_1, color='blue')
    plt.plot(r_peaks_mlii, signal_1[r_peaks_mlii], 'ro', markersize=5, label='R-peaks (MLII)')
    plt.xlabel('Frame ( 1/360 secondes)')
    plt.ylabel('Amplitude')
    plt.title('R peaks detection in MLII')
    plt.legend()
    plt.grid(True)
    plt.show()

    # visualisation of the r peaks in V5
    plt.figure(figsize=(14, 6))
    plt.plot(signal_2, color='green')
    plt.plot(r_peaks_v5, signal_2[r_peaks_v5], 'ro', markersize=5, label='R-peaks (V5)')
    plt.xlabel('Frame ( 1/360 secondes)')
    plt.ylabel('Amplitude')
    plt.title('R peaks detection in V5')
    plt.legend()
    plt.grid(True)
    plt.show()


# detect_and_visualize_r_peaks(105)
    

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
# Segmenting the ECG Signal (batches.....)

def detect_and_segment_r_peaks(Number_File, path='mitbih_database/'):
    path = path_to_csv(Number_File)
    df = pd.read_csv(path, delimiter=',')
    df.rename(columns = lambda x: x.replace("'","").replace(" ","_"), inplace=True)
    
    # Get the ECG signals from the specified columns
    samples = df[df.columns[0]]
    signal_1 = df[df.columns[1]]
    signal_2 = df[df.columns[2]]

    print(signal_1)

    # Detection of R-peaks for Signal 1
    r_peaks_mlii = ecg.ssf_segmenter(signal=signal_1, sampling_rate=1000.0, threshold=200, before=0.03, after=0.01)
    r_peaks_mlii = r_peaks_mlii['rpeaks']
    print(r_peaks_mlii)    

    # Detection of R-peaks for Signal 2
    r_peaks_v5 = ecg.ssf_segmenter(signal=signal_2, sampling_rate=1000.0, threshold=200, before=0.03, after=0.01)
    r_peaks_v5 = r_peaks_v5['rpeaks']
    
    # Define a window size for segmenting the signal
    window_size = 250  # Adjust as needed
    
    # Create segments around R-peaks for Signal 1
    segments1 = [signal_1[peak - window_size // 2: peak + window_size // 2] for peak in r_peaks_mlii]


    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
    p_q_segments1 = [signal_1[peak - window_size // 2: peak - window_size // 15] for peak in r_peaks_mlii]
    signal_test1 = p_q_segments1[2]
    p_peaks_mlii = ecg.ssf_segmenter(signal=signal_test1, sampling_rate=1000.0, threshold=70, before=0.03, after=0.01)
    p_peaks_mlii = (p_peaks_mlii['rpeaks']) + 583 
    """([signal_1[peak - window_size // 2]for peak in r_peaks_mlii])"""
    print(p_peaks_mlii) 
    plt.plot(signal_test1, color='blue')
    plt.plot(p_peaks_mlii, signal_test1[p_peaks_mlii], 'ro', markersize=5, label='P-peaks (MLII)')
    plt.show()

    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
    s_t_segments1 = [signal_1[peak + window_size // 15: peak + window_size // 2] for peak in r_peaks_mlii]
    signal_test2 = s_t_segments1[2]
    t_peaks_mlii = ecg.ssf_segmenter(signal=signal_test2, sampling_rate=1000.0, threshold=20, before=0.1, after=0.1)
    t_peaks_mlii = (t_peaks_mlii['rpeaks']) +  724 
    """([signal_1[peak + window_size // 2] for peak in r_peaks_mlii])"""
    print(t_peaks_mlii)
    plt.plot(signal_test2, color='blue')
    plt.plot(t_peaks_mlii, signal_test2[t_peaks_mlii], 'ro', markersize=5, label='T-peaks (MLII)')
    plt.show()

    print(segments1[2])
    print(p_q_segments1[2])
    print(s_t_segments1[2])

    pr_interval = abs(p_peaks_mlii - r_peaks_mlii[2])
    print("absolute(", p_peaks_mlii ,"-", r_peaks_mlii[2] ,") = " , pr_interval)
    print("pr interval is : pr_frame/360 = ", pr_interval/360)

    qt_interval = abs(t_peaks_mlii - r_peaks_mlii[2])
    print("absolute(",t_peaks_mlii ,"-", r_peaks_mlii[2],") = ",qt_interval)
    print("qt interval is : qt_frame = ", qt_interval/360) 

    # Create segments around R-peaks for Signal 2
    # segments2 = [ecg_signal2[peak - window_size // 2: peak + window_size // 2] for peak in r_peaks_v5]
    
    # Create a figure to display the ECG segments
    plt.figure(figsize=(14, 6))

    # Define the number of segments to display (maximum 7)
    num_segments = min(7, len(segments1))

    # Define the colors for the segments
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']

    # Define the position of separators (end of each segment)
    separator_positions = []

    seg = 2
    segment = segments1[seg]
    x_values = np.arange(len(segment)) + seg * len(segment)
    # print (x_values)
    plt.plot(x_values, segment, label=f'Segment {seg + 1}', color=colors[0])
    plt.show()

    p_q_segment = p_q_segments1[seg]
    p_q_x_values = np.arange(len(p_q_segment)) + seg * len(p_q_segment)
    # print (p_q_x_values)
    # plt.plot(p_q_x_values, p_q_segment, label=f'Segment {seg + 1}', color=colors[1])
    # plt.show()

    s_t_segment = s_t_segments1[seg]
    s_t_x_values = np.arange(len(s_t_segment)) + seg * len(s_t_segment)
    # print (s_t_x_values)
    # plt.plot(s_t_x_values, s_t_segment, label=f'Segment {seg + 1}', color=colors[2])
    # plt.show()



    # Plot the segments for Signal 1
    for i in range(num_segments):
        segment = segments1[i]
        x_values = np.arange(len(segment)) + i * len(segment)
        plt.plot(x_values, segment, label=f'Segment {i + 1}', color=colors[i])

        # Store the position of the separator (end of the segment)
        separator_positions.append(x_values[-1])


#     # Plot the segments for Signal 2
#     for i in range(num_segments):
#         segment = segments2[i]
#         x_values = np.arange(len(segment)) + (i + num_segments) * len(segment)
#         plt.plot(x_values, segment, label=f'Segment {i + 1}', color=colors[i])

#         # Store the position of the separator (end of the segment)
#         separator_positions.append(x_values[-1])

    # Add separators (vertical lines) between segments
    for pos in separator_positions:
        plt.axvline(x=pos, color='gray', linestyle='--', linewidth=1)

    # Customize labels and legend
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('ECG Segments with Separators Between Waves')
    plt.legend()
    plt.grid(True)
    plt.show()

detect_and_segment_r_peaks(105)
