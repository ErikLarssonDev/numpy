import numpy as np

# Checking numpy version
print(np.__version__)

a = np.array([1,2,3])
print(a)
print(a.shape)
print(a.dtype)
print(a.ndim)
print(a.size)
print(a.itemsize)
print(a[0])
a[0] = 10
print(a)

b = a * np.array([2,0,2]) # Pointwise multiplication
print(b)

l = [1,2,3]
a = np.array([1,2,3])
l.append(4) # Adds a 4 at the end of the list
l = l + [4] # Adds a 4 at the end of the list
l = l * 2 # Repeats list
print(l)
a = a + np.array([4]) # Adds 4 to each element in the numpy array (broadcasting)
a = a * 2 # Doubles each element
a = np.sqrt(a)
a = np.log(a)
print(a)

# Dot product
l1 = [1,2,3]
l2 = [4,5,6]

dot = 0
for i in range(len(l1)):
    dot += l1[i] * l2[i]
print(dot)

# Dot product with numy
a1 = np.array(l1)
a2 = np.array(l2)

dot = np.dot(a1,a2)
print(dot)

sum1 = a1 * a2
dot = np.sum(sum1)
print(dot)

dot = (a1 * a2).sum()
print(dot)

dot = a1 @ a2
print(dot)

# Speed test
from timeit import default_timer as timer
a = np.random.randn(1000)
b = np.random.randn(1000)

A = list(a)
B = list(b)

T = 1000

def dot1():
    dot = 0
    for i in range(len(A)):
        dot += A[i]*B[i]
    return dot
def dot2(): 
    return np.dot(a,b)

start = timer()
for t in range(T):
    dot1()
end = timer()
t1 = end-start

start = timer()
for t in range(T):
    dot2()
end = timer()
t2 = end-start

print('list calculation', t1)
print('np.dot', t2)
print('ratio', t1/t2)

# Multidimensional (nd) arrays
a = np.array([[1,2],[3,4]])
print(a)
print(a.shape)
print(a[0]) # Access first row
print(a[0][0]) # Access first element
print(a[0,0]) # Access first element
print(a[:,0]) # All rows, column 0
print(a.T) # Transpose array
print(np.linalg.inv(a)) # Calc inverse (needs a square array)
print(np.linalg.det(a)) # Calc determinant
print(np.diag(a)) # Diagonal elements
c = np.diag(a)
print(np.diag(c)) # Creates a diagonal matrix
print(a)
b = a[0,1]
print(b)
# Slicing
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)
b = a[0,:] # Row 0 all columns
b = a[-1,-2] # Last row, second last column
print(b)

a = np.array([[1,2],[3,4],[5,6]])
print(a)
bool_idx = a > 2
print(bool_idx)
print(a[bool_idx])
print(a[a > 2])
b = np.where(a > 2, a, -1)
print(b)

a = np.array([10,19,30,41,50,61])
print(a)
b = [1,3,5]
print(a[b]) # [19 41 61] (fancy indexing)

# Printing all even numbers
even = np.argwhere(a%2 == 0).flatten()
print(a[even])

a = np.arange(1,7) # Array with numbers 1-6
print(a)
b = a.reshape((2,3))
print(b)

a = np.arange(1,7)
b = a[np.newaxis,:] # Add dimension
print(b)

# Concatenation
a = np.array([[1,2], [3,4]])
b = np.array([[5,6]])
c = np.concatenate((a,b)) # Axis default 0
c = np.concatenate((a,b), axis=None)
c = np.concatenate((a,b.T), axis=1)
print(c)

# hstack
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
c = np.hstack((a,b))
print('hstack', c)

# vstack
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
c = np.vstack((a,b))
print('vstack', c)

# Broadcasting
x = np.array([[1,2,3], [4,5,6], [1,2,3], [4,5,6]])
a = np.array([1,0,1]) # Want to add this to all columns
y = x + a
print(y)

# Functions and axis
a = np.array([[7,8,9,10,11,12,13], [17,18,19,20,21,22,23]])
print(a)
print(a.sum()) # axis=None default
print(a.sum(axis=0)) # Sum for each column
print(a.sum(axis=1)) # Sum for each row

# Copying arrays
a = np.array([1,2,3])
b = a # Only copies the reference!
b = a.copy()
b[0] = 42
print(a)
print(b)

# Generating arrays
a = np.zeros((2,3))
print(a)

a = np.ones((2,3))
print(a)

a = np.full((2,3), 5.0)
print(a)

a = np.eye(3)
print(a)

a = np.arange(20)
print(a)

a = np.linspace(0,10,5)
print(a)

# Random numbers
a = np.random.random((3,2)) # Random numbers from the uniform dist 0-1
print(a)

a = np.random.randn(1000) # Gaussion (normal, mean=0, var=1)
print(a.mean())
print(a.var())

a = np.random.randint(3,10,size=(3,3)) # lower bound, upper bound, size
print(a)

a = np.random.choice(5, size=10) # Random 0-5
print(a)

a = np.random.choice([-8, -7, -6], size=10) # Random from the list
print(a)

# Linear algebra
a = np.array([[1,2], [3,4]])
eigenvalues, eigenvectors = np.linalg.eig(a)
print(eigenvalues)
print(eigenvectors) # Column vector!

# e_vec * e_val = A * e_vec
b = eigenvectors[:, 0] * eigenvalues[0]
print(b)

c = a @ eigenvectors[:, 0]
print(c)

# Don't use b==c to compare (numerical issues)
print(np.allclose(b,c))

# Solving linear systems
# Manual way
A = np.array([[1,1], [1.5,4.0]])
b = np.array([2200,5050])

x = np.linalg.inv(A).dot(b) # Inverse is quite slow, exists better way
print(x)

x = np.linalg.solve(A, b) # Preffered way
print(x)

# Load csv files
# np.loadtxt or np.genfromtxt
data = np.loadtxt('filename.csv', delimiter=",", dtype=np.float32) # Example only
