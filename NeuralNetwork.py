from utils import *
class MLP:
    def __init__(self, layers, learning_rate=0.01, bias=False):
        self.layers = layers
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = {}
        self.biases = {}
        self.activations = {}
        self.init_weights()
        np.random.seed(42)

    def init_weights(self):
        for i in range(1, len(self.layers)):
            input_units = self.layers[i - 1]['units']
            output_units = self.layers[i]['units']

            self.weights[i] = np.random.rand(input_units, output_units) 

            if self.bias:
                self.biases[i] = np.random.rand((1, output_units))

            self.activations[i] = self.layers[i]['activation']

    def forward_propagation(self, x):
        f_net = {0: x} 
        for i in range(1, len(self.layers)):
            z = np.dot(f_net[i - 1], self.weights[i])
            if self.bias:
                z += self.biases[i].flatten().T

            if callable(self.activations[i]):
                f_net[i] = self.activations[i](z)
            else:
                raise ValueError("Activation function must be a callable function")
        return f_net

    def backward_propagation(self, x, y, f_net):
        m = x.shape[0] 
        #print (f_net)
        grads = {}
        for i in reversed(range(1, len(self.layers))):
            if i == len(self.layers) - 1:
                # Output layer
                grads[i] =  y - f_net[i] 
                #print(f"Grads at output {grads[i]}")
                if callable(self.activations[i]):
                    grads[i] *= self.activations[i](f_net[i], derivative=True)
                else:
                    raise ValueError("Activation function must be a callable function")
                
            else:
                # Hidden layers
                grads[i] = np.dot(grads[i + 1], self.weights[i + 1].T)
                if callable(self.activations[i]):
                    grads[i] *= self.activations[i](f_net[i], derivative=True)
                else:
                    raise ValueError("Activation function must be a callable function")
        return grads
    def update_weights(self,grads,f_net):
        for i in range(1,len(self.layers)):
            self.weights[i] += self.learning_rate * np.outer(f_net[i-1] , grads[i])
            if self.bias:
                self.biases[i] += self.learning_rate  * np.sum(grads[i], axis=0, keepdims=True)

    def fit(self, x_train, y_train, epochs=100):
        if type(x_train) == pd.DataFrame:
            x_train = x_train.to_numpy()
        if type(y_train) == pd.DataFrame:
            y_train = y_train.to_numpy()
        for _ in range(epochs):
            for row,target in zip(x_train,y_train):
                f_net = self.forward_propagation(row)
                grads=self.backward_propagation(row, target, f_net)
                self.update_weights(grads,f_net)
            if _ % 10 == 0:
                mse=mean_squared_error(f_net[len(self.layers)-1],y_train)
                print(f"MSE For Epoch {_}: {round(mse,3)} \n")
       
    def predict(self, x_test):
        f_net = self.forward_propagation(x_test)
        output_layer = len(self.layers) - 1
        return f_net[output_layer]
    
    def get_acc(self,x,y):
        pred=self.predict(x)
        predictions_df = pd.DataFrame(pred, columns=sorted(list(y.unique())))
        predicted_labels = predictions_df.idxmax(axis=1)
        accuracy=sum(predicted_labels == y.reset_index(drop=True))
        print(f"Train Accuracy = {round(accuracy,3) / len(x)}")
        accuracy/=len(x)
        return accuracy

