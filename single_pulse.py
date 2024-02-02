import pandas as pd
import matplotlib.pyplot as plt

patient_100_file = "100.csv"
ecg100 = pd.read_csv(patient_100_file, index_col=0)

print(ecg100.head(20))

# ecg100["MLII"].plot()
# plt.title("30-minute EKG of Patient 100")

ecg100_1p = ecg100[1200:2500]["MLII"]

ecg100_1p.plot()
plt.title("First 4 heartbeats of patient 100's EKG")


plt.grid(True)
plt.show()