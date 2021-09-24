"""
    ALIOUSALAH Mohamed Nassim
"""


import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

"""
    To use your own image and see the output of your model, add it to the "/image folder, name it img.jpg and run the code"
"""

"""
    training set : 209 cat and non cat pic
    testing set : 50 cat and non cat pic
"""

def load_dataset():
    with h5py.File('datasets/catvnoncat/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('datasets/catvnoncat/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# Run this code to consult the training set (0 <= index <= 208)
"""
index = 88
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
"""

#Reshaping the training set examples


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Checking if the first 10 pixels of the second image are in the correct place
assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

#Standardizing the data

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


#sigmoid function
def sigmoid(z):
    return  (1 / (1 + np.exp(-z)))

#initializing the weight and the bias
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0
    return w, b

#propagate function that computes the cost function and its gradient (the "forward" and "backward" propagation steps for learning the parameters)
def propagate(w, b, X, Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)

    cost = -1/m * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))
    
    dw = 1/m * (np.dot(X, (A - Y).T))
    db = 1/m * np.sum(A - Y)

    return {"dw": dw,"db": db}, np.squeeze(np.array(cost))


#optimization function, This function optimizes w and b by running a gradient descent algorithm
#The goal is to learn  𝑤 and 𝑏 by minimizing the cost function  𝐽
#learning rules :
#                   w = w − 𝛼*𝑑w
#                   b = b − 𝛼*𝑑b
#where 𝛼 (alpha) is the learning rate

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y) 
       
        dw = grads["dw"]
        db = grads["db"]
        
        w -= learning_rate*dw
        b -= learning_rate*db
       
        if i % 100 == 0:
            costs.append(cost)
        
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    return {"w": w, "b": b}, {"dw": dw, "db": db}, costs

#with the optimization done, we can now use w and b with a 70% degree of certainty with the predict function
#predict Calculates  𝐴 = 𝜎(𝑤𝑇𝑋 + 𝑏), where 𝜎 is the sigmoid function
#Converts the entries of A into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector Y_prediction

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
    
    w, b = initialize_with_zeros(X_train.shape[0])
    
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params['w']
    b = params['b']
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#num_iterations and print_cost == 10000 and True respectively , but you can change them 
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=10000, learning_rate=0.005, print_cost=True)

"""
    Uncomment to take a look at the testing set results 
"""
#index = 8
#plt.imshow(test_set_x[:, index].reshape((64, 64, 3)))
#print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")


#Your image is here...
my_image = "img.jpg"   

#Preprocessing the image to fit the algorithm.
fname = "image/" + my_image
image = np.array(Image.open(fname).resize((64, 64)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, 64 * 64 * 3)).T


my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
