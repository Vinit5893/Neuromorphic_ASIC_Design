import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("GPU is not available.")

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))


def path_to_csv(file_number):
    filenumber = str(file_number) + '.csv'
    for dirname, _, filenames in os.walk('database/'):
        for filename in filenames:
            if (filename == filenumber):
                file_path = os.path.join(dirname, filename)
                # print(os.path.join(dirname, file_number))
                return file_path

path = path_to_csv('MIT-BIH Arrhythmia Database')
'''
# Reading MIT-BIH Arrhythmia Dataset as an example
data_df = pd.read_csv(path) 
print(data_df.shape)
print(data_df.head(20))

x_data = data_df.iloc[:, 2:]
y_label = data_df[['type']]

print(x_data.head(20))
print(y_label.value_counts())
'''

# Create your first MLP in Keras


# fix random seed for reproducibility
# np.random.seed(7)
# load pima indians dataset
dataset = pd.read_csv(path)
# split into input (X) and output (Y) variables
X = dataset.iloc[:,2:18]
Y = dataset[['type']]

Y.replace(['VEB', 'SVEB', 'F', 'Q'], [1,2,3,4], inplace=True)
Y.replace(['N'], 0, inplace=True)

# create model
model = Sequential()
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

from ann_visualizer.visualize import ann_viz
ann_viz(model, title="My graph")