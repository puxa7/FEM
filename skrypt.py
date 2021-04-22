import numpy as np
import matplotlib.pyplot as plt
from zadanie15 import function4
from scipy import integrate


# arr = np.array([1, 2, 3, 4, 5])
# print(arr)
# A = np.array([[1, 2, 3], [7, 8, 9]])
# print(A)
# A = np.array([[1, 2, \
#                 3],
#               [7, 8, 9]])
# print(A)

# v = np.arange(1,7)
# print(v,"\n")
# v = np.arange(-2,7)
# print(v,"\n")
# v = np.arange(1,10,3)
# print(v,"\n")
# v = np.arange(1,10.1,3)
# print(v,"\n")
# v = np.arange(1,11,3)
# print(v,"\n")
# v = np.arange(1,2,0.1)
# print(v,"\n")

# v = np.linspace(1,3,4)
# print(v)
# v = np.linspace(1,10,4)
# print(v)
# v = np.linspace(0,10,3)
# print(v)

# X = np.ones((2,3))
# Y = np.zeros((2,3,4))
# Z = np.eye(2) # np.eye(2,2) np.eye(2,3)
# Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))

# print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)
# U = np.block([[A], [X]])
# print(U)
# V = np.block([[
#     np.block([
#         np.block([[np.linspace(1,3,3)],[
#             np.zeros((2,3))]]) ,
#         np.ones((3,1))])
#     ],
#     [np.array([100, 3, 1/2, 0.333])]] )
# print(V)
# print( V[0,2] )
# print( V[3,0] )
# print( V[3,3] )
# print( V[-1,-1] )
# print( V[-4,-3] )
# print( V[3,:] )
# print( V[:,2] )
# print( V[3,0:3] )
# print( V[np.ix_([0,2,3],[0,-1])] )
# print( V[3] )

# Q = np.delete(V, 3, 0)
# print(Q)
# Q = np.delete(V, 2, 1)
# print(Q)
# v = np.arange(1,7)
# print(v)
# print( np.delete(v, 3, 0) )
# print(np.size(v))
# print(np.shape(v))
# print(np.size(V))
# print(np.shape(V))

# A = np.array([[1, 0, 0],
#               [2, 3, -1],
#               [0, 7, 2]] )

# B = np.array([[1, 2, 3],
#               [-1, 5, 2],
#               [2, 2, 2]] )
# print( A+B )
# print( A-B )
# print( A+2 )
# print( 2*A )

# MM1 = A@B
# print(MM1)
# MM2 = B@A
# print(MM2)

# MT1 = A*B
# print(MT1)
# MT2 = B*A
# print(MT2)

# C = np.linalg.solve(A,MM1)
# print(C) 
# x = np.ones((3,1))
# b =  A@x
# y = np.linalg.solve(A,b)
# print(y)

# PM = np.linalg.matrix_power(A,2) 
# print(PM)
# PT = A**2  
# print(PT)
# print(A.T)
# print(A.transpose())
# print(A.conj().T)
# print(A.conj().transpose())



# x=[1,2,3]
# y=[4,6,5]
# plt.plot(x,y)
# plt.show()

# x=np.arange(0.0, 2.0, 0.01)
# y1=np.sin(2.0*np.pi*x)
# y2=np.cos(2.0*np.pi*x)
# y=y1*y2
# l1, = plt.plot(x,y,'b:', linewidth = 3)
# l2,l3 = plt.plot(x,y1,'r*',x,y2,'g--',linewidth=3)
# plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres')
# plt.show()

# print(np.max(A))
# print(np.min(A))
# print(np.max(A,0))
# print(np.max(A,1))
# # ’wektoryzacja’ macierzy
# print( A.flatten() )
# print( A.flatten('F'))
# # wymiary macierzy
# print( np.shape(A) )
# # liczba elementow macierzy
# print( np.size(A) )

# A = np.array([[1,1,1],
# [1,1,0],
# [0,1,1]])
# b = np.array([[3],
# [2],
# [2]])
# x = np.linalg.solve(A, b)
# print(x)
# # wyznacznik macierzy
# print(np.linalg.det(A))
# # uwarunkowanie macierzy
# print(np.linalg.cond(A))
# # macierz odwrotna
# print(np.linalg.inv(A))



#6.3
X = np.block([[np.linspace(1,5,5)],[np.linspace(5,1,5)]])
Y = np.block([[2 * np.ones((2, 3))],[np.linspace(-90, -70, 3)]])
Z = np.block([np.zeros((3, 2)), Y])
P = np.block([[X],[Z]])
A = np.block([P, 10 * np.ones((5,1))])

print(A,"\n")

#6.4
B = A[1,:] + A[3,:]
print(B,"\n")

#6.5
C = np.array([])
C = np.append(C, max(A[:,0]))
C = np.append(C, max(A[:,1]))
C = np.append(C, max(A[:,2]))
C = np.append(C, max(A[:,3]))
C = np.append(C, max(A[:,4]))
C = np.append(C, max(A[:,5]))

print(C,"\n")

#6.6
D = np.array([])
D = np.delete(B, [0,5])

print(D,"\n")

#6.7
D[D == 4] = 0

print(D,"\n")

#6.8
E = np.array([])
L = np.min(C)
M = np.max(C)
E = np.delete(C, np.where(C == L))
E = np.delete(E, np.where(E == M))

print(E,"\n")


#6.9
L = np.min(C)
M = np.max(C)
for val in range(np.shape(A)[0]):
    if L in A[val,:]:
        if M in A[val,:]:
            print(A[val,:])

#6.10
print("tablicowo: ")
print(D*E,"\n")
print("wektorowo: ")
print(D@E,"\n")


#6.11
def function1(l):
    tab = np.random.randint(0,11,[l, l])
    return tab, np.trace(tab)

x = int(input('Podaj rozmiar: '))
print(function1(x),"\n")



#6.12
def function2(tab_1):
    np.fill_diagonal(tab_1, 0)
    np.fill_diagonal(np.fliplr(tab_1),0)
    return tab_1

tab_2 = np.array([[2,3,1,17],[5,7,8,16],[8,9,10,17],[11,12,13,17]])


print(function2(tab_2),"\n")

#6.13
def function3(tab_1):
    suma = 0
    size = np.shape(tab_1)
    for i in range(size[0]):
        if i % 2 == 0:
            suma = suma + np.sum(tab_1[i, :])
    return suma

print(function3(tab_2),"\n")

#6.14

x = np.linspace(-10, 10, 201)
y = lambda x: np.cos(2 * x)
plt.plot(x, y(x), 'r+')
plt.show()

#6.15

x = np.linspace(-10, 10, 201)
tab_4 = np.zeros([len(x)])

for i in range(len(x)):
    tab_4[i] = function4(x[i])

plt.plot(x,tab_4,'g+',x,y(x),'r--')

#6.17

y3=3*(y(x)) + tab_4
plt.plot(x,y3,'b*',x,tab_4,'g+',x,y(x),'r--')

#6.18

tab_5 = np.array([[10,5,1,7],[10,9,5,5],[1,6,7,3],[10,0,1,5]])
tab_6 = np.array([[34],[44],[25],[27]])

x= np.linalg.inv(tab_5) @ tab_6

print(x,"\n")


#6.19
x = np.linspace(0,2*np.pi,1000000)
y = lambda x:np.sin(x)

calka, error = integrate.quad(y, 0, 2*np.pi)
print(calka,"\n")


    
