import numpy as np 

mean = (1, 2, 0.5)
cov = [[1, 0, 0.5], [0, 1, 0.5], [0.5, 0, 0.75]]
x = np.random.multivariate_normal(mean, cov, (100))
val = np.zeros((100,1))
for i in range(100):
	if i%2==0:
		val[i,0] = 1
x = np.column_stack((x,val))
np.savetxt('train.csv', x, delimiter=',', newline='\n')
