import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


# ROOT_DIR = os.getcwd()
# path = os.path.join(ROOT_DIR, 'database\\INCART 2-lead Arrhythmia Database.csv') #database/INCART 2-lead Arrhythmia Database.csv   
# database/MIT-BIH Supraventricular Arrhythmia Database.csv
# database\\MIT-BIH Supraventricular Arrhythmia Database.csv

def path_fetch(file_name: str) -> str: 
    ROOT_DIR = os.getcwd()
    path_fetch = os.path.join(ROOT_DIR, file_name)
    return path_fetch

def extract_input_data(path: str, range_sel: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if range_sel == True:
        start_i, nos = range_select()
        print("Checking the range from %d to %d for %d number of samples" %(start_i, start_i + nos - 1, nos))
        X_new = pd.concat([
            df.iloc[start_i:start_i + nos, 2:4],
            df.iloc[start_i:start_i + nos, 9:13]
            ], axis=1)
        Y_new = df[['type']].iloc[start_i:start_i + nos]
        # print(X_new)
        # print(Y_new)

    else: 
        X_new = pd.concat([
            df.iloc[: , 2:4],
            df.iloc[: , 9:13]
            ], axis=1)
        Y_new = df[['type']]
        # print(X_new.head(20))
        # print(Y_new.head(20))

    return X_new, Y_new

def normalize_data(input: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 100))
    input_test_scaled = scaler.fit_transform(input)
    return input_test_scaled


def make_predictions(model: keras.models.Model, input_test_scaled: np.ndarray, Y_new: pd.DataFrame) -> list[int]:
    y_pred = model.predict(input_test_scaled)
    print(y_pred)
    predicted_class = np.argmax(y_pred, axis=1)
    print(predicted_class)
    return predicted_class

def range_select() -> tuple[int, int]:
    start_i = np.random.randint(0,100688)
    nos = np.random.randint(0,100)
    return start_i, nos


def print_predictions(predicted_class: list[int], Y_new: pd.DataFrame, range_sel: bool = False) -> None:
    class_mapping = {0: 'N', 1: 'VEB', 2: 'SVEB', 3: 'F', 4: 'Q'}
    predicted_classes = [class_mapping[int(pred)] for pred in predicted_class]
    y_actual_disp = Y_new['type'].tolist()
    y_actual_disp = [class_mapping[int(pred)] for pred in y_actual_disp]
    print("Predictions : Actual class")
    if range_sel == True:
        for pred, actual in zip(predicted_classes, y_actual_disp):
            print(pred.center(len("Predctions")+1), ":", actual.center(len("Actual class")+1))
    else:
        for pred, actual in zip(predicted_classes[:50], y_actual_disp[:50]):
            print(pred.center(len("Predctions")+1), ":", actual.center(len("Actual class")+1))


def plot_confusion_matrix(conf_matrix: np.ndarray) -> None:
    class_labels = ['N', 'VEB', 'SVEB', 'F', 'Q']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show() 

def metrix_calculation(confusion_matrix: np.ndarray) -> None:
    num_classes = len(confusion_matrix)
    TP = np.zeros(num_classes)
    TN = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)
    for i in range(num_classes):
        TP[i] = confusion_matrix[i, i]
        FP[i] = np.sum(confusion_matrix[:, i]) - TP[i]
        FN[i] = np.sum(confusion_matrix[i, :]) - TP[i]
        TN[i] = np.sum(confusion_matrix) - (TP[i] + FP[i] + FN[i])
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    for i in range(num_classes):
        print(f"Class {i}:")
        print(f"   Sensitivity  :  {sensitivity[i]}")
        print(f"   Specificity  :  {specificity[i]}")
        print(f"   Accuracy     :  {accuracy[i]}")


    
def main():
    sample_data_path = path_fetch('database\\MIT-BIH Arrhythmia Database copy.csv')
    trained_model_path = path_fetch('0.9731_model14_ECG_6_150_150_150_5.h5')
    range_sel = True
    X_new, Y_new = extract_input_data(sample_data_path, range_sel)
    model = load_model(trained_model_path)
    X_test_scaled = normalize_data(X_new)
    Y_new = Y_new.replace({'N': 0, 'VEB': 1, 'SVEB': 2, 'F': 3, 'Q': 4})
    predicted_class = make_predictions(model, X_test_scaled, Y_new)
    print_predictions(predicted_class, Y_new, range_sel)
    conf_matrix = confusion_matrix(Y_new, predicted_class)
    print(conf_matrix)
    metrix_calculation(conf_matrix)
    plot_confusion_matrix(conf_matrix)



if __name__ == "__main__":
    main()

















# predicted_class = np.argmax(y_pred, axis=1)
# print(predicted_class)
# print("The predicted class are :")

# class_mapping = {0: 'N', 1: 'VEB', 2: 'SVEB', 3: 'F', 4: 'Q'}

# # Replace the predicted integer classes with their string values
# predicted_classes = [class_mapping[pred] for pred in predicted_class]

# # Print the predicted classes vertically
# # for pred in predicted_classes:
# #     print(pred)

# # y_actual_disp = Y_new['type'].iloc[start_i:start_i + nos].tolist()
# y_actual_disp = Y_new['type'].iloc[:].tolist()
# print("Predictions : Actual class")

# for pred, actual in zip(predicted_classes, y_actual_disp):
#     print(pred.center(len("Predctions")+1), ":", actual.center(len("Actual class")+1))
# # print("Accuracy:",metrics.accuracy_score(y_val, y_pred))

# # print("*** Confusion Matrix ***")
# print(metrics.confusion_matrix(y_val, y_pred))