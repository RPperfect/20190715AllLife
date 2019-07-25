from numpy import *

def loadDataSet(fileName):
    dataMat=[]
	fr=open(fileName)
	for line in fr.readlines():
	    curLine=line.strip().split('\t')
		fltLine=map(float,curLine)
		dataMat.append(fltLine)
	return dataMat
	
def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:][0]
	mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:][0]
	return mat0,mat1
	
def regLeaf(dataSet):
    return mean(dataSet[:,-1])
	
def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]
	
def linearSolve(dataSet):
    m,n=shape(dataSet)
	X=mat(ones((m,n)))
	Y=mat(ones((m,1)))
	X[:,1:n]=dataSet[:,0:n-1]
	Y=dataSet[:,-1]
	if linalg.det(xTx)==0.0
	    raise NameError('you are wrong')
	ws=xTx.I*(X.T*Y)
	return ws,X,Y
	
def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
	return ws
	
def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
	yHat=X*ws
	return sum(power(Y-yHat,2))
	
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]
	tolN=ops[1]
	if len(set(dataSet[:,-1].T.tolist()[0]))==1:
	    return None,leafType(dataSet)
	m,n=shape(dataSet)
	S=errType(dataSet)
	bestS=inf
	bestIndex=0
	bestValue=0
	for featIndex in range(n-1):
	    for splitVal in set(dataSet[:,featIndex]):
		    mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
			if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
			    continue
			newS=errType(mat0)+errType(mat1)
			if newS<bestS:
			   bestIndex=featIndex
			   bestValue=splitVal
			   bestS=newS
	if(S-bestS)<tolS:
	    return None,leafType(dataSet)
	mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
	if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
	    return None,leafType(dataSet)
	return bestIndex,bestValue
	
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
