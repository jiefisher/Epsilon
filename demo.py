import numpy as np
import epsilon
X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], 
              [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
y = np.array([[1,0,0,0,0,0,0,1]]).T 
e=epsilon.Epsilon(X.shape[1])
e.add_layer(11)
e.add_layer(5)
e.add_layer(1)
e.fit(X,y)
print e.predict(X)
