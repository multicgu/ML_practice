import numpy as np
d = 0.85
N = 100
M = np.array([[0,0,0,1/3,0,0],[1/4,0,0,0,1/2,0],[0,1,0,1/3,1/2,0],[1/4,0,0,0,0,1],[1/4,0,1,1/3,0,0],[1/4,0,0,0,0,0]])
w = [1/6,1/6,1/6,1/6,1/6,1/6]

for i in range(N):
    w = np.dot(M,w)
node = ['a','b','c','d','e','f']
print("simple page rank:")
for i,n in zip(w,node):
    print("%s s PR is %3.2f" % (n,i))
    
w = [1/6,1/6,1/6,1/6,1/6,1/6]
for i in range(N):
    w = (1-d)/N + d*np.dot(M,w)

print("random walk page rank:")
for i,n in zip(w,node):
    print("%s s PR is %3.2f" % (n,i))
    
    
    
    
    
    