import pandas as pd
import numpy as np
import seaborn as sns
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as conf

def mean_squared_error(pred, actual):
    return np.round(np.sum(np.power((pred - actual), 2)) / (2 * len(actual)), 4)

def confusion_matrix(y_pred, y_true):
    labels = y_true.unique()
    print (labels) 
    num_labels = len(labels)
    cm = conf(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.show()
    return cm

def linear(x):
    return x

def sig_num(a):
    if a == 0:
        return -1
    return int(a / abs(a))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sigmoid(Z,derivative=False):
    sigmoid_value = 1 / (1 + np.exp(-Z))
    if derivative : 
        return Z * (1 - Z)
    else:
        return sigmoid_value

def tanh(Z, derivative=False):
    tanh_value = np.tanh(Z)
    if derivative : 
        return (1 - Z**2)
    else:
        return tanh_value



def remove_nulls(target,data:pd.DataFrame):
    for cat,df in data.groupby(target):
        for col in df.columns[0:-1]:
            df[col].fillna(df[col].mean(),inplace=True)
            data[data[target]==cat]=df
    return data

def split(target_col,df=pd.DataFrame()):        
    X = df.drop(columns=[target_col])
    Y = df[target_col]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    return X_train, Y_train, X_test, Y_test


def normalize_dataframe(df):
    normalized_df = pd.DataFrame()
    
    for col in df.columns[:-1]:
        min_val = df[col].min()
        max_val = df[col].max()
        normalized_col = (df[col] - min_val) / (max_val - min_val)
        normalized_df[col] = normalized_col
        
    normalized_df[df.columns[-1]] = df[df.columns[-1]]
    
    return normalized_df

def one_hot_encode_target(target_series):
    one_hot_encoded = pd.get_dummies(target_series, prefix='label')

    return one_hot_encoded

def label_encode_target(target_series):
    dummy_series=copy.deepcopy(target_series)
    for label,i in zip(dummy_series.unique(),range(0,len(dummy_series.unique()))) :
        dummy_series[dummy_series==label]=i
    return dummy_series.astype(float)

def generate_layers(num_layers, num_neurons,activation, train_df ):
    layers = {i: {"units": num_neurons, "activation": activation} for i in range(1,num_layers)}
    layers[0]={"units": train_df.shape[1] , "activation" : linear}
    layers[num_layers]={"units" : 3 , "activation" : activation}
    return layers
