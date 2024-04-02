import itertools
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#---------------------task1------------------------------------
#Load data
class Data_preprocessing:
    def load_dataset(self):
        data_path = "\Assessment1_Dataset.csv"
        self.dataset = pd.read_csv(data_path)


    def data_analysis(self):
        features = ["Amino_acid", "Malic_acid", "Ash", "Acl", "Mg", "Phenols", "Flavanoids", "Nonflavanoid_phenols",
                    "Proanth", "Colo_int", "Hue", "OD", "Proline"]


        plt.figure(figsize=(15, 10))

        # plot boxplot for each feature
        for i, feature in enumerate(features):
            plt.subplot(4, 4, i + 1)
            sns.boxplot(x=self.dataset[feature])
            plt.title(f"{feature} Distribution")


        plt.tight_layout()


        plt.show()

    def feature_importance(self):
        # compute the correlation matrix
        correlation_matrix = self.dataset.corr()

        # plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")

        # show the plot
        plt.show()

    def check_null(self):
        #Check
        null_values = self.dataset.isnull().sum()
        return null_values


    #split feature and target
    def split_data(self,test_size=0.2, validation_size = 0.25, random_state=0):
        #Split data
        data = self.dataset.copy(deep=True)
        X = data.drop("Producer", axis=1).values
        Y = data["Producer"].values
        X = self.normalize(X)
        # print(X)
        # print(X.shape)
        # print(Y.shape)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,random_state=random_state)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size,random_state=random_state)
        self.train_data = np.column_stack((x_train,y_train))
        self.test_data = np.column_stack((x_test,y_test))
        self.val_data = np.column_stack((x_val,y_val))
        return self.train_data,self.test_data,self.val_data

    def normalize(self, data_x):
        # Use L2 norm to normalize the data
        norm = np.linalg.norm(data_x, axis=0)
        data_x = data_x / norm
        return data_x

    #split data into train and test


    def process(self):
        self.load_dataset()
        # self.check_null()
        # self.data_analysis()
        # self.feature_importance()
        return self.split_data()


#---------------------task2------------------------------------

#loss function
#Considering the aim is to achieve classification task over 3 classes, softmax was used.

class FullyConnectedLayer:
    def __init__(self,input_num,output_num):
        np.random.seed(2)
        self.input_num = input_num
        self.output_num = output_num
        self.weight = np.random.randn(input_num,output_num)
        self.bias = np.zeros((1,output_num))

    def forward(self,input):
        self.input = input
        self.output = np.matmul(input,self.weight) + self.bias
        return self.output

    def backward(self,output_grad):
        self.d_weight = np.dot(self.input.T,output_grad)#dL/dw = X.T * dL/dy
        self.d_bias = np.sum(output_grad,axis=0) #dL/db = 1 + dL/dy
        self.d_input = np.dot(output_grad,self.weight.T)  #dL/dX = dL/dy*W.T
        return self.d_input

    def update_param(self,lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):  # load pre-trained parameters
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):  # save parameters
        return self.weight, self.bias



class ReLu:
    def __init__(self):
        print('use ReLu layer')

    def forward(self,input):
        self.input = input
        self.output = np.maximum(0,input)
        return self.output

    def backward(self,output_grad):
        self.d_input = output_grad * (self.input > 0)
        return self.d_input

    def ReLU_d(self,x):
        return np.where(x > 0,1,0)

class Softmax:
    def __init__(self,weights):
        print('use softmax layer')
        self.weights = weights


    def forward(self, prob):
        self.output = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)  #Matrix dimension batch_size*3

        return self.output

    def get_loss(self, label):  # 计算损失
        self.batch_size = self.output.shape[0]
        self.label = label
        self.label_onehot = np.zeros_like(self.output)
        label = label.flatten().astype(int)
        # print(self.label_onehot)
        # print(label)
        self.label_onehot[np.arange(self.batch_size), label-1] = 1.0 #onhot to represent the label(true value)
        entropy_loss = -np.sum(np.log(self.output) * self.label_onehot) / self.batch_size
        regularization = 0.001 * self.weights / (2*self.batch_size)
        total_loss = entropy_loss + regularization
        return total_loss
    def backward(self):  # 反向传播的计算
        self.output_grad = (self.output - self.label_onehot) / self.batch_size
        diff_softmax = (self.output - self.label_onehot) / self.batch_size #get deviation of output by chain rule
        return diff_softmax
class my_neural_network:
    def __init__(self, batch_size=30, input_size=13, hidden1=16, hidden2=32, hidden3=8, out_classes=3, lr=0.001, max_epoch=300,train_data=None,test_data=None,val_data=None):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.out_classes = out_classes
        self.max_epoch = max_epoch
        self.lr = lr
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data


    def shuffle_data(self):
        print('Randomly shuffle train data...')
        local_random = np.random.RandomState(123)
        local_random.shuffle(self.train_data)

    def build_model(self):
        # build 3 layer modal
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLu()

        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLu()

        self.fc3 = FullyConnectedLayer(self.hidden2, self.hidden3)
        self.relu3 = ReLu()

        self.fc4 = FullyConnectedLayer(self.hidden3, self.out_classes)
        weights = sum(sum(self.fc1.weight**2)) + sum(sum(self.fc2.weight**2)) +sum(sum(self.fc3.weight**2)) +sum(sum(self.fc4.weight**2))


        self.softmax = Softmax(weights)
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]


    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir,allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        self.fc4.load_param(params['w4'], params['b4'])

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        params['w4'], params['b4'] = self.fc4.save_param()
        np.save(param_dir, params)

    def forward(self, input):  # forward propagation
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        h3 = self.relu3.forward(h3)
        prob = self.fc4.forward(h3)
        prob = self.softmax.forward(prob)
        return prob

    def backward(self):  # backward propagation
        dloss = self.softmax.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.relu3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)
        return

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    # to create mini-batches
    def create_mini_batches(self, batch_size):
        mini_batches = []
        self.shuffle_data()
        data = self.train_data
        n_minibatches = data.shape[0] // batch_size
        # print(n_minibatches)
        i = 0
        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches
    def train(self):
        print('Start training...')
        self.train_losses = [] # reset train_losses and value_losses every train
        self.val_losses = []
        for epoch in range(self.max_epoch):
            batches = self.create_mini_batches(self.batch_size)
            for batch_x, batch_y in batches:
                self.forward(batch_x)
                train_loss = self.softmax.get_loss(batch_y)
                self.backward()
                self.update(self.lr)

                # Calculate validation loss at the end of each batch
                self.forward(self.val_data[:, :-1])
            val_loss = self.softmax.get_loss(self.val_data[:, -1])

                # Append train and validation losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Print loss at the end of each epoch
            print('Epoch %d, train loss: %.6f, val loss: %.6f' % (epoch, train_loss, val_loss))

    def evaluate(self,data):
        prob = self.forward(data[:,:-1])
        pred_labels = np.argmax(prob, axis=1)
        accuracy = np.mean(pred_labels+1 == data[:, -1])
        print("Predicted labels:",  pred_labels+1)
        print("True labels:     ",  data[:, -1].astype(int))
        print('Accuracy in test set: %f' % accuracy)
        return accuracy


    def plot_loss(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Loss vs Iteration lr: {self.lr} and batchsize: {self.batch_size}')
        plt.legend()
        plt.show()


def run_model(lr,batch_size):
    h1, h2, h3, e = 16, 32, 8, 3
    mlp = my_neural_network(hidden1=h1, hidden2=h2, hidden3=h3, max_epoch=500,batch_size=batch_size,lr=lr,train_data=train_data,
                            test_data=test_data,val_data=val_data)
    mlp.build_model()
    mlp.train()
    mlp.plot_loss()
    mlp.save_model('lr-%.3f-batch_size-%depoch.npy' % (lr, batch_size))
    #mlp.load_model('lr-%.3f-batch_size-%depoch.npy' % (lr, batch_size))
    return mlp


def grid_search(param_grid):

    val_accuracies = []  # 保存每次筛选的验证准确率
    for lr, batch_size in itertools.product(*param_grid.values()):
        mlp = run_model(lr, batch_size)  #
        val_accuracy = mlp.evaluate(val_data)  # get the validation accuracy
        val_accuracies.append((lr, batch_size, val_accuracy))  #
    return val_accuracies


def plot_validation_accuracies(val_accuracies):
    lr_values, batch_size_values, val_accuracy_values = zip(*val_accuracies)
    plt.figure(figsize=(10, 6))
    heatmap_data = []
    for lr, batch_size, accuracy in val_accuracies:
        heatmap_data.append([lr, batch_size, accuracy])

    heatmap_data.sort(key=lambda x: (x[0], x[1]))  # Sort by lr first, then batch_size

    heatmap_values = [[entry[2] for entry in heatmap_data if entry[0] == lr] for lr in
                      sorted(set([entry[0] for entry in heatmap_data]))]

    plt.imshow(heatmap_values, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Validation Accuracy')
    plt.xticks(range(len(set(batch_size_values))), sorted(set(batch_size_values)))
    plt.yticks(range(len(set(lr_values))), sorted(set(lr_values)))
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    plt.title('Validation Accuracy for Different LR and Batch Size Combinations')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def run_grid_search():
    # 定义要搜索的参数范围
    param_grid = {
        'lr': [0.001, 3e-3, 0.01, 0.03, 0.05],
        'batch_size': [20, 30, 40 , 50, 60]
    }
    # 执行网格搜索
    val_accuracies = grid_search(param_grid)
    print("Validation accuracies for different LR and Batch Size combinations:")
    for lr, batch_size, val_accuracy in val_accuracies:
        print(f"lr={lr}, batch_size={batch_size}: Validation Accuracy={val_accuracy}")
    # 绘制验证准确率图
    plot_validation_accuracies(val_accuracies)
def present_result():
    dp = Data_preprocessing()
    train_data, test_data, val_data = dp.process()
    present_result()
    mlp = run_model(0.01, 60)
    accuracy = mlp.evaluate(test_data)  # get the validation accuracy
if __name__ == "__main__":
    dp = Data_preprocessing()
    #
    train_data, test_data, val_data = dp.process()
    run_grid_search()

