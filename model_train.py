# %%
# !python.exe -m pip install --upgrade pip
# !pip install tensorflow
# !pip install scikit-learn
# !pip install keras
# !pip install seaborn
# !pip install ann_visualizer

# %%
import os
ROOT_DIR = os.getcwd()
print("Current working directory:", ROOT_DIR)


# %%
import tensorflow as tf

# if tf.config.list_physical_devices('GPU'):
#     print("GPU is available!")
# else:
#     print("GPU is not available.")


# %%
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
# from ann_visualizer.visualize import ann_viz

# %%
# import tensorflow as tf
# print("GPU Available:", tf.config.list_physical_devices('GPU'))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# def path_to_csv(file_number):
#     filenumber = str(file_number) + '.csv'
#     # for dirname, _, filenames in os.walk(''):
#     for filename in filenames:
#         if (filename == filenumber):
#             file_path = os.path.join(dirname, filename)
#             # print(os.path.join(dirname, file_number))
#             return file_path

# %%
path = 'MIT-BIH Arrhythmia Database copy.csv'

# %%
file_path  = os.path.join(ROOT_DIR, path)

# %%
print(file_path)

# %%
# fix random seed for reproducibility
# np.random.seed(7)

# %%
# load MIT-BIH dataset
dataset = pd.read_csv(file_path)

# %%
veb_rows = dataset[dataset['type'] == 'VEB'].copy()
sveb_rows = dataset[dataset['type'] == 'SVEB'].copy()
f_rows = dataset[dataset['type'] == 'F'].copy()
q_rows = dataset[dataset['type'] == 'Q'].copy()

# Append duplicated rows to the original DataFrame
dataset = pd.concat([dataset] + [veb_rows]*12 + [sveb_rows]*29 + [f_rows]*110 + [q_rows]*6200, ignore_index=True)

# Shuffle the DataFrame to mix the duplicated rows with the original ones
dataset = dataset.sample(frac=1).reset_index(drop=True)

# %%
# dataset

# %%
# split into input (X) and output (dataset) variables
# X = dataset.iloc[:,2:4,]
# X1 = dataset.iloc[:, 2:18]
X = pd.concat([
    dataset.iloc[:, 2:4],
    dataset.iloc[:, 9:13]
], axis=1)


Y = dataset[['type']]

# %%
# X

# %%
# dataset

# %%
# # normalize data
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X)

# dataset_normalized = pd.read_csv(scaler.fit_transform(path), columns=path.columns)

# %%

# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_scaled = min_max_scaler.fit_transform(X_train)
# X_test_scaled = min_max_scaler.transform(X_test)

# print(scaler.scale_)

# X_train_scaled

# %%
#Class Renaming
Y.replace(['N','VEB', 'SVEB', 'F', 'Q'], [0,1,2,3,4], inplace=True)

# %%
# Y

# %%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X , Y, test_size=0.2, random_state=42)

# %%
# X_train

# %%
# y_train

# %%
from matplotlib import pyplot as plt
Y['type'].plot(kind='line', figsize=(8, 4), title='type')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# %%
from matplotlib import pyplot as plt
Y['type'].plot(kind='hist', bins=20, title='type')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

# %%
# create model
model = Sequential()
model.add(Dense(6, input_dim=6, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(5, activation='softmax'))

# %%
import keras.backend as K

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_positives / (possible_negatives + true_positives)

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_positives + true_negatives)


# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[sensitivity, specificity, 'accuracy'])

# %%
# # Compile model
# from keras.optimizers import Adamax 
# # my_optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
# model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=2, batch_size=10,validation_data=(X_val, y_val))
# evaluate the model


# %%
# model

# %%
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# %%
# Save model
# model.save('model1_ECG.h5')

model_path = os.path.join(ROOT_DIR, '0.86_model5_ECG_6_20_20_5.h5')
model.save(model_path)

# %%
# from google.colab import files

# # Download the file
# files.download('model1_ECG.h5')

# %%
# Visualization
# from keras.layers import Dense
# from ann_visualizer.visualize import ann_viz

# ann_viz(model, title="My graph")

# %%
# path = os.path.join(ROOT_DIR, 'database\\MIT-BIH Arrhythmia Database copy.csv') #database/INCART 2-lead Arrhythmia Database.csvdatabase/MIT-BIH Supraventricular Arrhythmia Database.csv

# %%
import numpy as np
from keras.models import load_model
from sklearn import metrics

# Load the trained model
# file_path = os.path.join(ROOT_DIR, '0.86_model5_ECG_6_20_20_5.h5')
# model = load_model(file_path)

start_i = 80535
nos = 20

df = pd.read_csv(file_path)
X_new = pd.concat([
    df.iloc[start_i:start_i+nos, 2:4],
    df.iloc[start_i:start_i+nos, 9:13]
], axis=1)

Y_new = df[['type']]


# Get a section of the data from DataFrame Y
y_actual = Y_new[start_i:start_i+nos]


print(X_new)
print(y_actual)
# normalize data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_test_scaled = scaler.fit_transform(X_new)


# Make predictions
y_pred = model.predict(X_new)
# Print or use the predictions
print(y_pred)


predicted_class = np.argmax(y_pred, axis=1)
print(predicted_class)
print("The predicted class are :")

class_mapping = {0: 'N', 1: 'VEB', 2: 'SVEB', 3: 'F', 4: 'Q'}

# Replace the predicted integer classes with their string values
predicted_classes = [class_mapping[pred] for pred in predicted_class]

# Print the predicted classes vertically
for pred in predicted_classes:
    print(pred)

# print("Accuracy:",metrics.accuracy_score(y_val, y_pred))

# print("*** Confusion Matrix ***")
# print(metrics.confusion_matrix(y_val, y_pred))

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = y_actual  # True labels for the test data
y_pred = predicted_classes  # Predicted labels by the model

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix using heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# %%



