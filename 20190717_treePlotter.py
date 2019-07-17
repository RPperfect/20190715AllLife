import matplotlib.pyplot as plt

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs=0
	firstStr=myTree.keys()[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
	    if type(secondDict[key])._name_=='dict':
		   numLeafs+=getNumLeafs(secondDict[key])
		else:
		   numLeafs+=1
	return numLeafs
	
def getTreeDepth(myTree):
    maxDepth=0
	firstStr=myTree.keys()[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
	    if type(secondDict[key])._name_=='dict':
		   thisDepth=1+getTreeDepth(secondDict[key])
		else:
		   thisDepth=1
		if thisDepth>maxDepth;
		   maxDepth=thisDepth
	return maxDepth
	
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
	
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
	yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
	
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
	depth=getTreeDepth(myTree)
	firstStr=myTree.keys()[0]
	cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
	plotMidText(cntrPt,parentPt,nodeTxt)
	plotNode(firstStr,cntrPt,parentPt,decisionNode)
	secondDict=myTree[firstStr]
	plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
	for key in secondDict.keys():
