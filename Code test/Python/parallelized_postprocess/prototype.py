
import time as ti
import numpy as np
import multiprocessing
import sys



#==========================================
def myfunction(x,y,z):
    
    ti.sleep(1)
    
    return 3*x, y**2, z+1

#==========================================
#Parallelized computation
def wrapper(inputs) :
    
    func = inputs[0]
    args = inputs[1:]
    
    return func(*args)

#==========================================   
#parallelizes the calculation of a function 
#needs : the inputs data as a single array, the first element must be the object function to parallelize
#        the lenght of the axis along which to iterate, the nbr of cores to use
#        the arrays where to put the results  
    
def parallel(inputs, lenght, nbrCores, *outputs):
    
    workers = multiprocessing.Pool(processes = nbrCores)
    results = workers.imap(wrapper,inputs)
    
    for i, result in enumerate(results):     
        for j, output in enumerate(outputs): 
            
            if len(result)!=len(outputs): output[i] = result
            else:                         output[i] = result[j]
            
        sys.stderr.write('\rProgress : {0}/{1}'.format(i+1,lenght))
         
    workers.close()
    workers.join() 
    workers.terminate()
    
    print ""

    return 0



#==========================================
#inputs
x = np.ones([10,100])
y = np.ones([10,100])
z = np.ones([10,100])
N = len(x)

multiplied_x = np.zeros([10,100])
squared_y = np.zeros([10,100])
added_z = np.zeros([10,100])

nbrCores = 2

inputs = (np.array([myfunction, x[i], y[i], z[i]]) for i in xrange(N))

strt = ti.time()
parallel(inputs, N, nbrCores, multiplied_x, squared_y, added_z)
print ti.time()-strt















