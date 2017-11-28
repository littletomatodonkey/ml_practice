# -*- coding: utf-8 -*-
from numpy import  *

arr = random.rand(4, 4)
print (arr)

rndMat = mat(arr)
print ("\r\n")
print ( rndMat.A )

res = rndMat.A * rndMat.I
print ( res )
