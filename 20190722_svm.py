from numpy import *
from time import sleep

def loadDataSet(fileName):#加载文件，取出第一列和第二列的特征数据，取出第三列的标签数据
    dataMat=[]
	labelMat=[]
	fr=open(fileName)
	for line in fr.readlines():
	    lineArr=line.strip().split('\t')
		dataMat.append(float(lineArr[0]),float(lineArr[1]))
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat
	
def selectJrand(i,m):#保证选择的j不等于i
    j=i
	while(j==i)
	    j=int(random.uniform(0,m))
	return j
	
def clipAlpha(aj,H,L):#控制L<=aj<=H
    if aj>H:
	   aj=H
	if L>aj:
	   aj=L
	return aj
	
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn)
	labelMat=mat(classLabels).transpose()
	b=0
	m,n=shape(dataMatrix)
	alphas=mat(zeros(m,1))
	iter=0
	while(iter<maxIter):
	    alphaPairsChanged=0
		for i in range(m):
		    fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
			Ei=fXi-float(labelMat[i])
			if ((labelMat[i]*Ei<-toler)and(alphas[i]<C)) or ((labelMat[i]*Ei>toler)and(alphas[i]>0)):
                            j=selectJrand(i,m)
                            fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                            Ej=fXj-float(labelMat[j])
                            alphaIold=alphas[i].copy()
                            alphaJold=alphas[j].copy()
                            if(labelMat[i]!=labelMat[j]):
                                L=max(0,alphas[j]-alphas[i])
                                H=min(C,C+alphas[j]-alphas[i])
                            else:
                                L=max(0,alphas[j]+alphas[i]-C)
                                H=min(C,alphas[j]+alphas[i])
                            if L==H:
                                print"L==H"
                                continue
                            eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                            if eta>=0:
                    print"eta>=0"
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
			        print"j not moving enough"
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T		
	        b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
	            b=b1
		elif (0<alphas[j]) and (C>alphas[j]):
		    b=b2
		else:
		    b=(b1+b2)/2.0
		alphaPairsChanged+=1
		print('iter:%d i:%d,pairs changed %d' %(iter,i,alphaPairsChanged))
	if(alphaPairsChanged==0):
		    iter+=1
		else:
		    iter=0
		print('iteration number: %d' %iter)
	return b,alphas
	
def kernelTrans(X,A,kTup):
    m,n=shape(X)
	K=mat(zeros((m,1)))
	if kTup[0]=='lin':
	    K=X*A.T
	elif kTup[0]=='rbf':
	    for j in range(m):
		    deltaRow=X[j,:]-A
			K[j]=deltaRow*deltaRow.T
		K=exp(K/(-1*kTup[1]**2))
	else:
	    raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
	return K
	
class optStruct:
    def _init_(self,dataMatIn,classLabels,C,toler,kTup):
	    self.X=dataMatIn
		self.labelMat=classLabels
		self.C=C
		self.tol=toler
		self.m=shape(dataMatIn)[0]
		self.alphas=mat(zeros((self.m,self.m)))
		for i in range(self.m):
		    self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)
			
def calcEk(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
	Ek=fXk-float(oS.labelMat[k])
	return Ek
	
def selectJ(i,oS,Ei):
    maxK=-1
	maxDeltaE=0
	Ej=0
	oS.eCache[i]=[1,Ei]
	validEcacheList=nonzero(oS.eCache[:,0].A)[0]
	if (len(validEcacheList))>1:
	    for k in validEcacheList:
		    if k==i:
			    continue
			Ek=calcEk(oS,k)
			deltaE=abs(Ei-Ek)
			if(deltaE>maxDeltaE):
			    maxK=k
				maxDeltaE=deltaE
				Ej=Ek
		return maxK,Ej
	else:
	     j=selectJrand(i,oS.m)
		 Ej=calcEk(oS,j)
	return j,Ej
	
def updateEk(oS,k):
    Ek=calcEk(oS,k)
	oS.eCache[k]=[1,Ek]
