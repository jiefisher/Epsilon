import numpy as np
def sigmod(M):
    return 1/(1+np.exp(-M))
class Epsilon(object):
    def __init__(self,shape):
        self.w=[]
        self.x_shape=shape
    def add_layer(self,num):
        if len(self.w)==0:
            num1=self.x_shape+1
        else:
            num1=self.w[len(self.w)-1].shape[1]
        num2=num
        self.w.append(2*np.random.random((num1,num2))-1)
    def fit(self,x,y,alpha=1,epoch=5000):
        bias=np.ones((x.shape[0],1))
        x=np.hstack((x,bias))
        h=[0 for i in range(len(self.w)+1)]
        h[0]=x
        for j in range(epoch):
            for i in range(len(self.w)):
                h[i+1]=sigmod(np.dot(h[i],self.w[i]))
            error=h[len(h)-1]-y
            for i in range(len(self.w),0,-1):
                delta=error*h[i]*(1-h[i])
                error=delta.dot(self.w[i-1].T)
                self.w[i-1]+=(-1)*alpha*h[i-1].T.dot(delta)
    def predict(self,x):
        bias=np.ones((x.shape[0],1))
        x=np.hstack((x,bias))
        for w in self.w:
            x=sigmod(np.dot(x,w))
        return x

