import numpy as np
import os
from keras.models import load_model
import pandas as pd
from sklearn import metrics
import keras
import keras.backend as K
import numpy as np

ROOT_DIR = os.getcwd()
path = os.path.join(ROOT_DIR, 'database\\MIT-BIH Arrhythmia Database copy.csv') #database/INCART 2-lead Arrhythmia Database.csv   
# database/MIT-BIH Supraventricular Arrhythmia Database.csv
# database\\MIT-BIH Supraventricular Arrhythmia Database.csv

# Load the trained model
# file_path = os.path.join(ROOT_DIR, '0.86_model5_ECG_6_20_20_5.h5')
file_path = os.path.join(ROOT_DIR, '0.9662_model13_ECG_6_50_50_50_5.h5')

# with keras.utils.custom_object_scope({'sensitivity': sensitivity, 'specificity': specificity}):
model = load_model(file_path)

start_i = np.random.randint(0,100688)
nos = np.random.randint(0,100)

df = pd.read_csv(path)
X_new = pd.concat([
    df.iloc[: , 2:4],
    df.iloc[: , 9:13]
    # df.iloc[start_i:start_i + nos, 2:4],
    # df.iloc[start_i:start_i + nos, 9:13]
], axis=1)

Y_new = df[['type']]


# Get a section of the data from DataFrame Y
y_actual = Y_new[:]
# y_actual = Y_new[start_i:start_i + nos]

print("Checking the range from %d to %d for %d number of samples" %(start_i, start_i + nos - 1, nos))
print(X_new)
# print(y_actual)


# normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 100))
X_test_scaled = scaler.fit_transform(X_new)


# Make predictions
y_pred = model.predict(X_test_scaled)
# Print or use the predictions
print(y_pred)


predicted_class = np.argmax(y_pred, axis=1)
print(predicted_class)
print("The predicted class are :")

class_mapping = {0: 'N', 1: 'VEB', 2: 'SVEB', 3: 'F', 4: 'Q'}

# Replace the predicted integer classes with their string values
predicted_classes = [class_mapping[pred] for pred in predicted_class]

# Print the predicted classes vertically
# for pred in predicted_classes:
#     print(pred)

y_actual_disp = Y_new['type'].iloc[start_i:start_i + nos].tolist()
print("Predictions : Actual class")

for pred, actual in zip(predicted_classes, y_actual_disp):
    print(pred.center(len("Predctions")+1), ":", actual.center(len("Actual class")+1))
# print("Accuracy:",metrics.accuracy_score(y_val, y_pred))

# print("*** Confusion Matrix ***")
# print(metrics.confusion_matrix(y_val, y_pred))
