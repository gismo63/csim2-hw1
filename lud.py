import numpy as np
from scipy import linalg
import sys

def LU_solve(A,b):
    r = 0
    size = len(A)
    L = np.eye(size)
    while r<(size-1):
        if A[r][r]!=0:
            for i in range(r+1,size):
                if A[i][r]!=0:
                    L_c = L_mat(r,i,size,A[i][r]/A[r][r])
                    A = np.dot(L_c,A)
                    L = np.dot(L_c,L)
            r+=1
        else:
            ind = first_nonzero(A[:,r],r,size)
            if ind == -1:
                return ("unavoidable zero pivot")
            P = perm_mat(ind,r,size)

            A = np.dot(P,A)
            L = np.dot(P,L)
    if abs(A[size-1][size-1])<10.0**(-10):
        return ("unavoidable zero pivot")
    c = np.dot(L,b)
    x = back_subs(A,c,size)
    return x


def L_mat(r,i,size,factor):
    L = np.eye(size)
    L
    L[i][r] =  -factor
    return L

def first_nonzero(col,r,size):
    for i in range(r,size):
        if col[i]!=0:
            return i
    return -1

def perm_mat(i,r,size):
    P = np.eye(size)
    P[i][i] = 0
    P[r][r] = 0
    P[r][i] = 1
    P[i][r] = 1
    return P

def forward_subs(L,b,size):
    x = np.zeros(size)
    x[0] = b[0]/L[0][0]
    for i in range(1,size):
        sum = 0
        for j in range(i):
            sum += L[i][j]*x[j]
        x[i] = (b[i]-sum)/L[i][i]
    return x


def back_subs(U,b,size):
    x = np.zeros(size)
    x[size-1] = b[size-1]/U[size-1][size-1]
    for i in range(1,size):
        sum = 0
        for j in range(i):
            sum += U[size-1-i][size-1-j]*x[size-1-j]
        x[size-1-i] = (b[size-1-i]-sum)/U[size-1-i][size-1-i]
    return x


A = np.array([[4,1,1,2],[1,6,1,4],[1,1,12,5],[2,4,5,14]])
b = np.array([5,-2,6,9])

print (LU_solve(A,b))
lu, p = linalg.lu_factor(A)
print(linalg.lu_solve((lu, p), b, True))
print (linalg.solve(A,b))
print ("\n")

A = np.array([[1,1,12,5],[1,3,1,4],[4,1,1,5],[2,4,5,7]])
b = np.array([5,-2,6,9])

print (LU_solve(A,b))
lu, p = linalg.lu_factor(A)
print(linalg.lu_solve((lu, p), b, True))
print (linalg.solve(A,b))
print ("\n")

A = np.array([[2,1,1,2],[1,3,1,4],[1,1,6,5],[2,4,7,9]])
b = np.array([5,-2,6,9])

print (LU_solve(A,b))
lu, p = linalg.lu_factor(A)
print(linalg.lu_solve((lu, p), b, True))
print (linalg.solve(A,b))
print ("\n")

A = np.array([[0,1,1,2],[0,1,1,4],[0,7,2,6],[1,1,7,9]])
b = np.array([5,-2,6,9])

print (LU_solve(A,b))
lu, p = linalg.lu_factor(A)
print(linalg.lu_solve((lu, p), b, True))
print (linalg.solve(A,b))
print ("\n")
