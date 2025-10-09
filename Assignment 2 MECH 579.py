import numpy as np
import matplotlib as plot

#Rosenbrock Function
def f(x,y):
  return (1-x)**2+100*(y-x**2)**2


#Steepest Descent
#General form: xk+1=xk+ alpha*p

#Initialize
x=np.array([1.2],[1])
alpha=1
rho=

#Gradient of f(x,y)
def grad_f(x,y):
  return np.array([2*x-2-400*x*y+400*x**3],[200*y-200*x**2])

#Search direction p
p=-1*grad_f(x,y)

#Steph length alpha (Using simple backtracking line search) 
#Start with alpha=1 and decrease