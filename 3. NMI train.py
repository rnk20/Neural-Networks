import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import os

def read_dataset():
    df = pd.read_csv(os.getcwd() + "/dataset_40_sonar.csv")
    X = df[df.columns[0:60]].values
    Y = df[df.columns[60]]
    encode = LabelEncoder()
    encode.fit(Y)
    Y = encode.transform(Y)
    Y = one_hot_encode(Y)
    return (X,Y)

def one_hot_encode(labels):
    n_labels=len(labels)
    n_unique_labels=len(np.unique(labels))
    one_hot_encode=np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels]=1
    return one_hot_encode

X, Y=read_dataset()
X, Y=shuffle(X, Y,random_state=1)
train_x,test_x,train_y,test_y = train_test_split(X, Y,test_size=0.20,random_state=415)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(Y.shape)


# Important parameter and variables to work with tensors
learning_rate=0.3
training_epochs=1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim= ",n_dim)
n_class = 2
model_path =os.getcwd() + "SAVENMI/NMI"

# Define the number of layers and number of neurons for each layer

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32,[None,n_dim])
y_ = tf.placeholder(tf.float32,[None,n_class])
w = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros(n_class))


weigths = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

#define our model
def multilayer_perceptron(x,weigths,biases):
    
    layer_1 = tf.add(tf.matmul(x, weigths['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weigths['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weigths['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    layer_4 = tf.add(tf.matmul(layer_3, weigths['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    out_layer = tf.matmul(layer_4, weigths['out']) + biases['out']
    return out_layer

init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = multilayer_perceptron(x,weigths,biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict = {x:train_x, y_:train_y})
    cost = sess.run(cost_function, feed_dict = {x:train_x, y_:train_y})
    cost_history = np.append(cost_history, cost)
    
    correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    accuracy = sess.run(accuracy, feed_dict = {x: train_x, y_:train_y})
    accuracy_history.append(accuracy)
    
    pred_y = sess.run(y, feed_dict={x:test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    
    
    print('epoch= ',epoch,' cost=',cost,' mse=',mse_,' Train_accuracy=',accuracy)

save_path = saver.save(sess,model_path)

print('Model Saved in file : ',save_path)
