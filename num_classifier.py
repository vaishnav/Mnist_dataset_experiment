#importing libraries
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from os import getcwd
from sklearn.linear_model import LogisticRegression
from time import sleep
from random import randint
from sklearn.metrics import accuracy_score



filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

#ADD HERE THE PATH WHERE YOU'VE DOWNLODED THE DATASET
datapath = '../../'


#code


def init():
    k=0
    mnist={}
    while k == 0:                  #for taking manual files or load from existing pickle datastructure
        ch = input("read from file or take input from old prestored data structure ")
        if ch == '1':
            k=2
            mnist = read()
        elif ch == '2':
            k=2
            mnist = inp()
        else:
            print("wrong choice try again \n")
            sleep(2)
    

    model,accuracy = testing(mnist)
    print("accuracy is ",accuracy)
    sleep(2)
    k=0
    while k ==0:
        ch = input("1 for testing random digit from test set \n2 to quit\n")
        if ch == '1':
            testing_random(model,mnist)
        else:
            k=1


#TRAINIG THE MODEL
def testing(mnist):
    clf = LogisticRegression(max_iter = 6)    #as after multiple tries it was the best and most suitable value
    clf.fit(mnist['training_images'],mnist['training_labels'])
    prediction=clf.predict(mnist['test_images'])
    acc = accuracy_score(mnist['test_labels'],prediction)
    return([clf,acc])


#TESTING RANDOM IMAGES FROM TEST IMAGES DATASET
def testing_random(model,mnist):
    #loading a random training image and displaying on screen
    ran=randint(0,mnist['test_images'].shape[0])
    plt.imshow(mnist['test_images'][ran,:].reshape(28,28))
    print("actual label ",mnist['test_labels'][ran])
    print("predicted label ",model.predict(mnist['test_images'][ran,:].reshape(1,-1)))   #reshape as model takes only 2d array input
    plt.show()



#READING THE DATASET AND STORING IT IN A PICKLE FILE
def read():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(datapath+name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(datapath+name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(getcwd()+"mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Data structure updated")
    sleep(2)
    return mnist


#LOADING THE DATASET FROM PICKLE FILE
def inp():
    with open(getcwd()+"/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
        print("Data structure loded")
        sleep(2)
    return mnist


if __name__ == '__main__':
    init()

















