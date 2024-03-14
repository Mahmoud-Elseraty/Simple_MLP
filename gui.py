import tkinter as tk
from tkinter import messagebox
from utils import *
from NeuralNetwork import MLP
from DataPrep import x_train, y_train, x_test, y_test, y_train_encoded

model=None

def train_network(hidden_layers,neurons_in_layers,learning_rate,epochs,use_bias,activation_function):
    layers=generate_layers(train_df=x_train
                    ,num_layers=hidden_layers + 1 , num_neurons= neurons_in_layers ,activation=globals()[activation_function.lower()])
    mlp=MLP(layers=layers,learning_rate= learning_rate)
    mlp.fit(x_train=x_train,y_train=y_train_encoded,epochs=epochs)
    acc=mlp.get_acc(x=x_train,y=y_train) 
    return mlp,acc

def test_sample(model:MLP,y_test,x_test):
    predictions=model.predict(x_test=x_test)
    predictions_df = pd.DataFrame(predictions, columns=sorted(list(y_test.unique())))
    predicted_labels = predictions_df.idxmax(axis=1)
    cm=confusion_matrix(y_pred=predicted_labels,y_true=y_test)
    accuracy=cm.trace() / len(y_test)
    messagebox.showinfo(f"Testing Completed", "Neural Network Testing Completed Successfully! \n Test Accuracy = {}".format(round(accuracy,3)))


def create_nn_gui():
    root = tk.Tk()
    root.title("Neural Network Configuration")
    def start_training():
        # User inputs from the entries
        global model
        params={}
        params["hidden_layers"] = int(hidden_layers_entry.get())
        params["neurons_in_layers"] = int(neuron_entry.get())
        params["learning_rate"] = float(learning_rate_entry.get())
        params["epochs"] = int(epochs_entry.get())
        params["use_bias"] = use_bias_var.get()
        params["activation_function"] = activation_var.get()
        print(params)
        # Training Start
        model,train_acc=train_network(**params)  
        messagebox.showinfo(f"Training Complete", "Neural Network Training Completed Successfully! \n Train Accuracy = {}".format(train_acc))

    def test():
        global model
        if not isinstance(model,MLP) : 
            raise AttributeError("Cannot Test Before Training")
        test_sample(model=model,y_test=y_test,x_test=x_test)  # Implement this function

    labels = ['Number of hidden layers:', 'Neurons in each hidden layer:', 'Learning rate (eta):',
              'Number of epochs (m):', 'Add bias:', 'Activation function:']

    for i, label_text in enumerate(labels):
        label = tk.Label(root, text=label_text)
        label.grid(row=i, column=0, padx=10, pady=5)
        if label_text=='Number of hidden layers:':
            hidden_layers_entry=tk.Entry(root)
            hidden_layers_entry.grid(row=i, column=1, padx=10, pady=5)
            hidden_layers_entry.insert(0,"1")
        elif label_text == 'Neurons in each hidden layer:':
            neuron_entry = tk.Entry(root)
            neuron_entry.grid(row=i, column=1, padx=10, pady=5)
            neuron_entry.insert(0,"3")
        elif label_text == 'Learning rate (eta):':
            learning_rate_entry=tk.Entry(root)
            learning_rate_entry.grid(row=i, column=1, padx=10, pady=5)
            learning_rate_entry.insert(0,"0.1")
        elif label_text == "Number of epochs (m):":
            epochs_entry=tk.Entry(root)
            epochs_entry.grid(row=i, column=1, padx=10, pady=5)
            epochs_entry.insert(0,100)
        elif label_text == 'Add bias:':
            use_bias_var = tk.BooleanVar()
            use_bias_check = tk.Checkbutton(root, variable=use_bias_var)
            use_bias_check.grid(row=i, column=1, padx=10, pady=5)
        elif label_text == 'Activation function:':
            activation_var = tk.StringVar()
            activation_var.set("Sigmoid")  # Default selection
            activation_options = ["Sigmoid", "Tanh"]
            activation_menu = tk.OptionMenu(root, activation_var, *activation_options)
            activation_menu.grid(row=i, column=1, padx=10, pady=5)
        else:
            entry = tk.Entry(root)
            entry.grid(row=i, column=1, padx=10, pady=5)

    train_button = tk.Button(root, text="Start Training", command=start_training)
    train_button.grid(row=len(labels) + 1, column=0, columnspan=2, padx=10, pady=10)

    classify_button = tk.Button(root, text="Test", command=test)
    classify_button.grid(row=len(labels) + 2, column=0, columnspan=2, padx=10, pady=10)

    root.mainloop()

