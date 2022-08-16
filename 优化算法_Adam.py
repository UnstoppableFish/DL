import numpy as np
import matplotlib.pyplot as plt
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

def cost(Y,Ahat):
    n = Y.shape[1]
    cost = (-1 / n) * np.sum(Y * np.log(Ahat) + (1 - Y) * np.log(1 - Ahat))
    cost = np.squeeze(cost)
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

def back_second(dA,activation,caechs,Y):
    A_caech,Z_caech = caechs
    if activation == 'relu':
        dz = grad_relu(dA,Z_caech)
        dw,db,dA_ = back_first(dz,A_caech,Y)
    else:
        dz = grad_sigmoid(dA,Z_caech)
        dw,db,dA_ = back_first(dz,A_caech,Y)
    return dw,db,dA_

def back_third(Ahat,Y,caechs,Net):
    Y = Y.reshape(Ahat.shape)
    dAl = - np.divide(Y, Ahat) + np.divide(1 - Y, 1 - Ahat)
    m = len(Net)
    grad={}
    grad['dA'+str(m)] = dAl
    now_caech = caechs[m - 1]
    grad['dw'+str(m)],grad['db'+str(m)],grad['dA'+str(m-1)] = back_second(dAl,'sigmoid',now_caech,Y)
    for i in reversed(range(1,m)):
        now_caech = caechs[i-1]
        grad['dw' + str(i)], grad['db' + str(i)], grad['dA'+str(i-1)] = back_second(grad['dA'+str(i)], 'relu', now_caech, Y)
    return grad

def update(Net,grads,parameters,beta1,beta2,t):
    learning_rate = 0.002
    epsilon = 1e-8
    L = len(parameters) // 2  # 整除
    v = {}
    s = {}
    ep = 1e-8
    for l in range(1,L+1):
        v["dw" + str(l)] = np.zeros_like(parameters["w" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s['dw'+str(l)] = np.zeros_like(parameters['w'+str(l)])
        s['db'+str(l)] = np.zeros(parameters['b'+str(l)].shape)
    for l in range(1,L+1):
        L = len(parameters) // 2
        v_corrected = {}  # 偏差修正后的值
        s_corrected = {}  # 偏差修正后的值

    for l in range(L):
        # 梯度的移动平均值,输入："v , grads , beta1",输出：" v "
        v["dw" + str(l + 1)] = beta1 * v["dw" + str(l + 1)] + (1 - beta1) * grads["dw" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # 计算第一阶段的偏差修正后的估计值，输入"v , beta1 , t" , 输出："v_corrected"
        v_corrected["dw" + str(l + 1)] = v["dw" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        # 计算平方梯度的移动平均值，输入："s, grads , beta2"，输出："s"
        s["dw" + str(l + 1)] = beta2 * s["dw" + str(l + 1)] + (1 - beta2) * np.square(grads["dw" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])

        # 计算第二阶段的偏差修正后的估计值，输入："s , beta2 , t"，输出："s_corrected"
        s_corrected["dw" + str(l + 1)] = s["dw" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))


        parameters["w" + str(l + 1)] = parameters["w" + str(l + 1)] - learning_rate * (v_corrected["dw" + str(l + 1)] / np.sqrt(s_corrected["dw" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))


    return parameters
# def test_update(X,Net,Y):
#     paras = para(X,Net)
#     Ahat,caechs = pro(X,paras)
#     grad = back_third(Ahat,Y,caechs,Net)
#     paras = update(Net,grad,paras,0.9,0.999,2)
#     grad = back_third(Ahat,Y,caechs,Net)
#     paras = update(Net,grad,paras,0.9,0.999,3)
#     return paras
# a = test_update(train_X,[4,1],train_Y)
# print(a)

def learning(X,Y,Net,num):
    np.random.seed(3)
    paras = para(X,Net)
    costs =[]
    for i in range(num):
        Ahat,caechs = pro(X,paras)
        costa = cost(Y,Ahat)
        costs.append(costa)
        grad = back_third(Ahat,Y,caechs,Net)
        paras = update(Net,grad,paras,0.9,0.9,num)
        if i%50==0:
            print('the cost is %s' % (costa))
    return paras,costs

def pred(paras,X,Y):
    Ahat,caechs = pro(X,paras)
    Y_pred = np.zeros((1,Y.shape[1]))
    for i in range(Y.shape[1]):
        if Ahat[0,i]>=0.5:
            Y_pred[0,i] = 1
        else:
            Y_pred[0,i] = 0
    print("准确度为: " + str(float(np.sum((Y_pred == Y)) / Y.shape[1])))

paras,co = learning(train_X,train_Y,[20,4,1],2500)
pred(paras,train_X,train_Y)
pred(paras,test_X,test_Y)
plt.plot(co)
plt.title('cost')
plt.xlabel('num')
plt.ylabel('loss')
plt.show()
