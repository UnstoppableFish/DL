import numpy as np
import matplotlib.pyplot as plt
# import h5py
from lr_utils import load_dataset

train_X,train_Y,test_X,test_Y,classs=load_dataset()
train_X=train_X.reshape(train_X.shape[0],-1).T
test_X=test_X.reshape(test_X.shape[0],-1).T
train_X=train_X/255
test_X=test_X/255


def sigmoid(z):
    Zcaech = z
    A=1/(1+np.exp(-z))
    return A,Zcaech

def relu(z):
    A=np.maximum(0,z)
    Zcaech = z
    return A,Zcaech

def grad_sigmoid(dA,Z_caech):
    z=Z_caech
    A=1/(1+np.exp(-z))
    S=dA*A*(1-A)
    return S

def grad_relu(dA,Z_caech):
    z=Z_caech
    A=np.array(dA,copy=True)
    A[z<=0] = 0
    return A

def para(X,Net):
    parameters = {}
    m = len(Net)
    np.random.seed(3)
    parameters['w'+str(1)] = np.random.randn(Net[0],X.shape[0])*0.01
    parameters['b'+str(1)] = np.zeros((Net[0],1))
    for i in range(1,m):
        parameters['w'+str(i+1)] = np.random.randn(Net[i],Net[i-1])*0.01
        parameters['b'+str(i+1)] = np.zeros((Net[i],1))
    return parameters

def pro_first(w,b,A_):
    z = np.dot(w,A_)+b
    A_caech = (A_,w,b)
    return z,A_caech

def pro_second(A_,w,b,activation):
    if activation == 'relu':
        z,A_caech= pro_first(w,b,A_)
        A,Zcaech = relu(z)
    else:
        z,A_caech = pro_first(w,b,A_)
        A,Zcaech = sigmoid(z)
    total_caech = (A_caech,Zcaech)
    return  A,total_caech

def pro(X,para):
    caechs=[]
    m = len(para)//2
    A_ = X
    for i in range(1,m):
        A,caech = pro_second(A_,para['w'+str(i)],para['b'+str(i)],'relu')
        A_ = A
        caechs.append(caech)
    Ahat,caech = pro_second(A_,para['w'+str(m)],para['b'+str(m)],'sigmoid')
    caechs.append(caech)

    return Ahat,caechs
def keep_pro(X,para,keep_pro):
    keep_pro = 1
    caechs=[]
    m = len(para)//2
    D = {}
    A_ = X
    for i in range(1,m):
        A,caech = pro_second(A_,para['w'+str(i)],para['b'+str(i)],'relu')
        A_ = A
        D['D'+str(i)] = np.random.randn(A_.shape[0], A_.shape[1])
        D['D'+str(i)] = D['D'+str(i)] <= keep_pro
        A_ = A_ * D['D'+str(i)]
        A_ = A_ / keep_pro
        caechs.append(caech)
    A,caech = pro_second(A_,para['w'+str(m)],para['b'+str(m)],'sigmoid')
    A_ = A
    D['D' + str(m)] = np.random.randn(A_.shape[0], A_.shape[1])
    D['D' + str(m)] = D['D' + str(m)] <= keep_pro
    A_ = A_ * D['D' + str(m)]
    Ahat = A_ / keep_pro
    caechs.append(caech)

    return Ahat,caechs,D

def cost(Y,Ahat):
    n = Y.shape[1]
    cost = (-1 / n) * np.sum(Y * np.log(Ahat) + (1 - Y) * np.log(1 - Ahat))
    cost = np.squeeze(cost)
    return cost
def re_cost(Y,Ahat,landa,para):
    n = Y.shape[1]
    cost = -(1 / n) * np.sum(Y * np.log(Ahat) + (1 - Y) * np.log(1 - Ahat))
    m = len(para)//2
    regulation = landa*(np.sum(np.square(para['w'+str(1)])) + np.sum(np.square(para['w'+str(2)])))
    cost = cost + regulation
    return cost
# def test_pro(X,Net):
#     paras = para(X,Net)
#     Ahat,caechs = pro(X,paras)
#     print(Ahat)
# test_pro(train_X,[4,4,1])

def back_first(dz,caech,Y):
    m = Y.shape[1]
    A_,w,b = caech
    dw = (1/m)*np.dot(dz,A_.T)
    db = (1/m)*np.sum(dz,axis=1,keepdims=True)
    dA_ = np.dot(w.T,dz)
    return dw,db,dA_

def keep_back_first(dz,caech,Y):
    m = Y.shape[1]
    A_,w,b = caech
    dw = (1/m)*np.dot(dz,A_.T)
    db = (1/m)*np.sum(dz,axis=1,keepdims=True)
    dA_ = np.dot(w.T,dz)
    return dw,db,dA_

def re_back_first(dz,caech,Y,lanbd,w):
    m = Y.shape[1]
    A_,w,b = caech
    dw = (1/m)*np.dot(dz,A_.T)+((lanbd*w)/m)
    db = (1/m)*np.sum(dz,axis=1,keepdims=True)
    dA_ = np.dot(w.T,dz)
    return dw,db,dA_

def back_second(dA,activation,caechs,Y):
    A_caech,Z_caech = caechs
    if activation == 'relu':
        dz = grad_relu(dA,Z_caech)
        dw,db,dA_ = back_first(dz,A_caech,Y)
    else:
        dz = grad_sigmoid(dA,Z_caech)
        dw,db,dA_ = back_first(dz,A_caech,Y)
    return dw,db,dA_

def keep_back_second(dA,activation,caechs,Y):
    A_caech,Z_caech = caechs
    if activation == 'relu':
        dz = grad_relu(dA,Z_caech)
        dw,db,dA_ = keep_back_first(dz,A_caech,Y)
    else:
        dz = grad_sigmoid(dA,Z_caech)
        dw,db,dA_ = keep_back_first(dz,A_caech,Y)
    return dw,db,dA_


def back_third(Ahat,Y,caechs,Net):
    Y = Y.reshape(Ahat.shape)
    dAl = -np.divide(Y,Ahat)+np.divide(1-Y,1-Ahat)
    m = len(Net)
    grad={}
    now_caech = caechs[m - 1]
    grad['dw'+str(m)],grad['db'+str(m)],grad['dA'+str(m)] = back_second(dAl,'sigmoid',now_caech,Y)
    for i in reversed(range(m-1)):
        now_caech = caechs[i]
        grad['dw' + str(i+1)], grad['db' + str(i+1)], grad['dA'+str(i+1)] = back_second(grad['dA'+str(i+2)], 'relu', now_caech, Y)
    return grad

def keep_back_third(Ahat,Y,caechs,Net,D):
    keep_prb = 1
    Y = Y.reshape(Ahat.shape)
    grad={}
    m = len(Net)
    dAl = -np.divide(Y,Ahat)+np.divide(1-Y,1-Ahat)
    grad['dA'+str(m)] = dAl

    dAl = dAl * D['D'+str(m)]
    dAl = dAl / keep_prb

    now_caech = caechs[m - 1]
    grad['dw'+str(m)],grad['db'+str(m)],grad['dA'+str(m-1)] = keep_back_second(dAl,'sigmoid',now_caech,Y)
    for i in reversed(range(1,m)):
        now_caech = caechs[i-1]
        grad['dA'+str(i)] = grad['dA'+str(i)] * D['D'+str(i)]
        grad['dA' + str(i)] = grad['dA'+str(i)] / keep_prb
        grad['dw' + str(i)], grad['db' + str(i)], grad['dA'+str(i-1)] = keep_back_second(grad['dA'+str(i)], 'relu', now_caech, Y)
    return grad

def update(Net,grads,parameters):
    learning_rate = 0.00001
    L = len(parameters) // 2  # 整除
    for l in range(1,L+1):
        parameters["w" + str(l)] = parameters["w" + str(l)] - learning_rate * grads["dw" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters
# def test_update(X,Net,Y):
#     paras = para(X,Net)
#     Ahat,caechs,cost = pro(X,Y,paras)
#     grad = back_third(Ahat,Y,caechs,Net)
#     paras = update(X,Net,grad,paras)
#     grad = back_third(Ahat,Y,caechs,Net)
#     paras = update(X,Net,grad,paras)
#     return paras
# a = test_update(train_X,[4,4,1],train_Y).keys()
# print(a)

def learning(X,Y,Net,num):
    np.random.seed(3)
    paras = para(X,Net)
    co=[]
    for i in range(num):
        Ahat,caechs = pro(X,paras)

        costs = cost(Y,Ahat)
        co.append(costs)
        grad = back_third(Ahat,Y,caechs,Net)
        paras = update(Net,grad,paras)
        if i%50==0:
            print('the cost is %s' % (costs))
    return paras,co

def keep_learning(X,Y,Net,num,keep_prb,):
    np.random.seed(3)
    paras = para(X,Net)
    co=[]
    for i in range(num):
        Ahat,caechs,D = keep_pro(X,paras,1)
        costs = cost(Y,Ahat)
        co.append(costs)
        grad = keep_back_third(Ahat,Y,caechs,Net,D)
        paras = update(Net,grad,paras)
        if i%50==0:
            print('the cost is %s' % (costs))
    return paras,co

def pred(paras,X,Y):
    Ahat,caechs = pro(X,paras)
    Y_pred = np.zeros((1,Y.shape[1]))
    for i in range(Y.shape[1]):
        if Ahat[0,i]>=0.5:
            Y_pred[0,i] = 1
        else:
            Y_pred[0,i] = 0
    print("准确度为: " + str(float(np.sum((Y_pred == Y)) / Y.shape[1])))

paras,co = keep_learning(train_X,train_Y,[25,20,5,1],1500,1)
pred(paras,train_X,train_Y)
plt.plot(co)
plt.title('cost')
plt.xlabel('num')
plt.ylabel('loss')
plt.show()
