#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt


# In[2]:


# read data into memory
#THE CODE TAKES ALMOST 1.5 MINUTES TO RUN
image_set = np.genfromtxt("hw06_images.csv", delimiter = ",")
label_set = np.genfromtxt("hw06_labels.csv", delimiter = ",").astype(int)
y_c = np.genfromtxt("hw06_labels.csv", delimiter = ",").astype(int)

# get X and y values
x_train = image_set[0:1000][:]
x_test = image_set[1000:][:]
y_train = label_set[0:1000]
y_test = label_set[1000:]



# get number of samples and number of features
N_train = len(y_train)
D_train = x_train.shape[1]


# In[3]:


class_y = np.zeros((1000,5))
                   

for i in range(0,len(y_train)):
    class_y[i][y_train[i]-1]=1
    
    
    
    
class_y=2*np.array(class_y)-1


# In[4]:


def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)


# In[5]:


s = 10
K_train = gaussian_kernel(x_train, x_train, s)
yyK = np.matmul(class_y[:,0][:,None], class_y[:,0][None,:]) * K_train

# set learning parameters
C = 10
epsilon = 1e-3

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * class_y[:,0][None,:])
b = cvx.matrix(0.0)
                    
# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha1 = np.reshape(result["x"], N_train)
alpha1[alpha1 < C * epsilon] = 0
alpha1[alpha1 > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha1 != 0)
active_indices, = np.where(np.logical_and(alpha1 != 0, alpha1 < C))
w01 = np.mean(class_y[:,0][active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha1[support_indices])))
f1_predicted = np.matmul(K_train, class_y[:,0][:,None] * alpha1[:,None]) + w01


# In[6]:


s = 10
K_train = gaussian_kernel(x_train, x_train, s)
yyK = np.matmul(class_y[:,1][:,None], class_y[:,1][None,:]) * K_train

# set learning parameters
C = 10
epsilon = 1e-3

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * class_y[:,1][None,:])
b = cvx.matrix(0.0)
                    
# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha2 = np.reshape(result["x"], N_train)
alpha2[alpha2 < C * epsilon] = 0
alpha2[alpha2 > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha2 != 0)
active_indices, = np.where(np.logical_and(alpha2 != 0, alpha2 < C))
w02 = np.mean(class_y[:,1][active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha2[support_indices])))


f2_predicted = np.matmul(K_train, class_y[:,1][:,None] * alpha2[:,None]) + w02


# calculate confusion matrix


# In[7]:


s = 10
K_train = gaussian_kernel(x_train, x_train, s)
yyK = np.matmul(class_y[:,2][:,None], class_y[:,2][None,:]) * K_train

# set learning parameters
C = 10
epsilon = 1e-3

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * class_y[:,2][None,:])
b = cvx.matrix(0.0)
                    
# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha3 = np.reshape(result["x"], N_train)
alpha3[alpha3 < C * epsilon] = 0
alpha3[alpha3 > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha3 != 0)
active_indices, = np.where(np.logical_and(alpha3 != 0, alpha3 < C))
w03 = np.mean(class_y[:,2][active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha3[support_indices])))


f3_predicted = np.matmul(K_train, class_y[:,2][:,None] * alpha3[:,None]) + w03


# calculate confusion matrix


# In[8]:


s = 10
K_train = gaussian_kernel(x_train, x_train, s)
yyK = np.matmul(class_y[:,3][:,None], class_y[:,3][None,:]) * K_train

# set learning parameters
C = 10
epsilon = 1e-3

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * class_y[:,3][None,:])
b = cvx.matrix(0.0)
                    
# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha4 = np.reshape(result["x"], N_train)
alpha4[alpha4 < C * epsilon] = 0
alpha4[alpha4 > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha4 != 0)
active_indices, = np.where(np.logical_and(alpha4 != 0, alpha4 < C))
w04 = np.mean(class_y[:,3][active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha4[support_indices])))


f4_predicted = np.matmul(K_train, class_y[:,3][:,None] * alpha4[:,None]) + w04


# calculate confusion matrix


# In[9]:


s = 10
K_train = gaussian_kernel(x_train, x_train, s)
yyK = np.matmul(class_y[:,4][:,None], class_y[:,4][None,:]) * K_train

# set learning parameters
C = 10
epsilon = 1e-3

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * class_y[:,4][None,:])
b = cvx.matrix(0.0)
                    
# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha5 = np.reshape(result["x"], N_train)
alpha5[alpha5 < C * epsilon] = 0
alpha5[alpha5 > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha5 != 0)
active_indices, = np.where(np.logical_and(alpha5 != 0, alpha5 < C))
w05 = np.mean(class_y[:,4][active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha5[support_indices])))


f5_predicted = np.matmul(K_train, class_y[:,4][:,None] * alpha5[:,None]) + w05



# In[10]:


f_all = np.zeros((1000,1))

f_pred = np.hstack((f1_predicted,f2_predicted,f3_predicted,f4_predicted,f5_predicted))

    
    
for i in range(0,1000):
    f_all[i]=int(np.argmax(f_pred[i])+1)
    
    
    
confusion_matrix = pd.crosstab(np.reshape(f_all, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[11]:


#TRAINING ENDS TESTING STARTS


# In[12]:


s = 10
K_test = gaussian_kernel(x_test, x_train, s)


test_f1_predicted = np.matmul(K_test, class_y[:,0][:,None] * alpha1[:,None]) + w01
test_f2_predicted = np.matmul(K_test, class_y[:,1][:,None] * alpha2[:,None]) + w02
test_f3_predicted = np.matmul(K_test, class_y[:,2][:,None]* alpha3[:,None]) + w03
test_f4_predicted = np.matmul(K_test, class_y[:,3][:,None]* alpha4[:,None]) + w04
test_f5_predicted = np.matmul(K_test, class_y[:,4][:,None]* alpha5[:,None]) + w05



# In[13]:


f_test_all = np.zeros((4000,1))

f_test_pred = np.hstack((test_f1_predicted,test_f2_predicted,test_f3_predicted,test_f4_predicted,test_f5_predicted))

    
    
for i in range(0,4000):
    f_test_all[i]=int(np.argmax(f_test_pred[i])+1)
    
    
    


# In[14]:


test_confusion_matrix = pd.crosstab(np.reshape(f_test_all, len(y_test)), y_test, rownames = ['y_predicted'], colnames = ['y_test'])
print(test_confusion_matrix)
    


# In[15]:


def alpha_pred(y_training,x_training,s1,c1,N_train):


    s = s1
    K_train = gaussian_kernel(x_training, x_training, s1)
    yyK = np.matmul(y_training[:,None], y_training[None,:]) * K_train

    # set learning parameters
    C = c1
    epsilon = 1e-3

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_training[None,:])
    b = cvx.matrix(0.0)
                    
# use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_training[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))


    ff_predicted = np.matmul(K_train, y_training[:,None] * alpha[:,None]) + w0
    
    
    
    return [ff_predicted,alpha,w0]
# calculate confusion matrix


# In[16]:


def SupportVector(y_full,x_training,s1,c1,N_train,y_train):
    
    
    f1,a1,w1 = alpha_pred(y_full[:,0],x_training,s1,c1,N_train)
    f2,a2,w2 = alpha_pred(y_full[:,1],x_training,s1,c1,N_train)
    f3,a3,w3 = alpha_pred(y_full[:,2],x_training,s1,c1,N_train)
    f4,a4,w4 = alpha_pred(y_full[:,3],x_training,s1,c1,N_train)
    f5,a5,w5 = alpha_pred(y_full[:,4],x_training,s1,c1,N_train)
    
    
    ff = np.zeros((N_train,1))

    f_predict = np.hstack((f1,f2,f3,f4,f5))

    a_list = np.array([a1,a2,a3,a4,a5])
    w_list = np.array([w1,w2,w3,w4,w5])
    for i in range(0,N_train):
        ff[i]=int(np.argmax(f_predict[i])+1)
    
        
    #tcm = pd.crosstab(np.reshape(ff, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
    
    
    
    
    
    return [ff,a_list,w_list]
    


# In[17]:


c_set = np.array([0.1,1,10,100,1000])

train_list=[]
test_list=[]
alpha_list=[]
w0_list=[]

for C in c_set:
    f,a,w=SupportVector(class_y,x_train,10,C,len(y_train),y_train)
    train_list.append(f)
    alpha_list.append(a)
    w0_list.append(w)

    
    




# In[18]:


train_accuracy=[]
for i in range(5):
    l1=train_list[i]-y_train[:,None]
    l1[l1!=0]=1
    l1=np.array(l1)
    train_accuracy.append(1-(np.sum(l1)/len(l1)))
   
    


# In[19]:


print(train_accuracy)


# In[20]:


alpha_list=np.array(alpha_list)
alpha_list=alpha_list.reshape((5,5,1000))


# In[21]:



w0_list=np.array(w0_list)
w0_list=w0_list.reshape((5,5))



fin_test_list=[]

for i in range(0,5):
    f_den1 = np.matmul(K_test, class_y[:,0][:,None] * alpha_list[i][0][:,None]) + w0_list[i][0]
    f_den2 = np.matmul(K_test, class_y[:,1][:,None] * alpha_list[i][1][:,None]) + w0_list[i][1]
    f_den3 = np.matmul(K_test, class_y[:,2][:,None] * alpha_list[i][2][:,None]) + w0_list[i][2]
    f_den4 = np.matmul(K_test, class_y[:,3][:,None] * alpha_list[i][3][:,None]) + w0_list[i][3]
    f_den5 = np.matmul(K_test, class_y[:,4][:,None] * alpha_list[i][4][:,None]) + w0_list[i][4]
    
    c_test_all = np.zeros((4000,1))

    c_test_pred = np.hstack((f_den1,f_den2,f_den3,f_den4,f_den5))

    
    
    for i in range(0,4000):
        c_test_all[i]=int(np.argmax(c_test_pred[i])+1)
    fin_test_list.append(c_test_all)
    


# In[22]:


fin_test_list=np.array(fin_test_list)

test_accuracy=[]
for i in range(5):
    l2= fin_test_list[i,:]-y_test[:,None]
    l2[l2!=0]=1
    l2=np.array(l2)
    
    test_accuracy.append(1-(np.sum(l2)/len(l2)))
   


# In[23]:


print(test_accuracy)


# In[24]:





x_ax=c_set
y_ax1=train_accuracy
plt.xlabel("Regularization Parameter")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.plot(x_ax,y_ax1,color="blue",marker = "." ,markersize=10,label="train")
y_ax2=test_accuracy
plt.plot(x_ax,y_ax2,color="red",marker = ".",markersize=10,label="test")
plt.legend(loc="upper left")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




