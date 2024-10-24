import numpy as np
class LinearReg:
  def __init__(self,lr=0.01,epochs=1000):
    self.lr=lr
    self.weights=None
    self.epochs=epochs
    self.bais=0

  def fit(self,X,y):
    if X.ndim==1:
      X=X.reshape(-1,1)
    N,n_features=X.shape
    self.weights=np.zeros(n_features)
    self.bais=0

    for _ in range(self.epochs):
      y_pred=np.dot(X,self.weights)+self.bais
      dw=(1/N)+np.dot(X.T,(y_pred-y))
      db=(1/N)+np.sum((y_pred-y))
      self.weights=self.weights-self.lr*dw
      self.bais=self.bais-self.lr*db
  def predict(self,X):
    if X.ndim==1:
      X=X.reshape(-1,1)
    return np.dot(X,self.weights)+self.bais

def rmse(y_test,ypred):
  return np.sqrt(np.mean((y_test-ypred)**2))


