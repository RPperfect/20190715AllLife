class treeNode:
    def _init_(self,nameValue,numOccur,parentNode):
	    self.name=nameValue
		self.count=numOccur
		self.nodeLink=None
		self.parent=parentNode
		self.children={}
	
	def inc(self,numOccur):
	    self.count+=numOccur
		
	def disp(self,ind=1):
	    print(' '*ind,self.name,' ',self.count)
		for child in self.children.values():
		    child.disp(ind+1)
			
def createTree(dataSet,minSup=1):
    headerTable={}
	for trans in dataSet:
	    for item in trans:
		    headerTable[item]=headerTable.get(item,0)+dataSet[trans]
	for k in headerTable.keys():
	    if headerTable[k]<minSup:
		    del(headerTable[k])
	freqItemSet=set(headerTable.keys())
	if len(freqItemSet)==0:
	    return None,None
	for k in headerTable:
	    headerTable[k]=[headerTable[k],None]
	retTree=treeNode('Null Set',1,None)
	for tranSet,count in dataSet.items():
	    localD={}
		for item in tranSet:
		    if item in freqItemSet:
			    localD[item]=headerTable[item][0]
		if len(localD)>0:
		    orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
			updateTree(orderedItems,retTree,headerTable,count)
	return retTree,headerTable
	
def updateTree(items,inTree,headerTable,count):
    if items[0] in inTree.children:
	    inTree.children[item[0]].inc(count)
	else:
	    inTree.children[item[0]]=treeNode(items[0],count,inTree)
		if headerTable[items[0]][1]==None:
		    headerTable[items[0]][1]=inTree.children[items[0]]
		else:
		    updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
	if len(items)>1:
	    updateTree(items[1::],inTree.children[items[0]],headerTable,count)
		
def updateHeader(nodeToTest,targetNode):
    while(nodeToTest.nodeLink!=None):
	    nodeToTest=nodeToTest.nodeLink
	nodeToTest.nodeLink=targetNode
	
def ascendTree(leadNode,prefixPath):
    if leafNode.parent!=None:
	    prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):
    condPats={}
    while treeNode!=None:
        prefixPath=[]
        ascendTree(treeNode,prefixPath)
        if len(prefixPath)>1:
            condPats[frozenset(prefixPath[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return condPats

def mineTree(inTree,headerTable,minSup,preFix,freqItemSet):
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[1])]
    for basePat in bigL:
        newFreqSet=preFix.copy()
        newFreqSet=add(basePat)
        freqItemSet.append(newFreqSet)
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])
        myCondTree,myHead=createTree(condPattBases,minSup)
        if myHead!=None:
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemSet)

def loadSimpDat():
    simpDat=[['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):	
    retDict={}
	for trans in dataSet:
	    retDict[frozenset(trans)]+=1
	return retDict
	
import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved=re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*','',bigString)
	listOfTokens=re.split(r'\W*',urlsRemoved)
	return [tok.lower() for tok in listOfTokens if len(tok)>2]
	
def getLotsOfTweets(searchStr):
    CONSUMER_KEY=''
	CONSUMER_SECRET=''
	ACCESS_TOKEN_KEY=''
	ACCESS_TOKEN_SECRET=''
	api=twitter.Api(consumer_key=CONSUMER_KEY,consumer_secret=CONSUMER_SECRET,access_token_key=ACCESS_TOKEN_KEY,access_token_secret=ACCESS_TOKEN_SECRET)
	resultsPages=[]
	for i in range(1,15):
	    print('featching page %d' %i)
		searchResults=api.GetSearch(searchStr,per_page=100,page=i)
		resultsPages.append(searchResults)
		sleep(6)
	return resultsPages
	
def mineTweets(tweetArr,minSup=5):
    parsedList=[]
	for i in range(14):
	    for j in range(100):
		    parsedList.append(textParse(tweetArr[i][j].text))
	initSet=createInitSet(parsedList)
	myFPtree,myHeaderTab=createTree(initSet,minSup)
	myFreqList=[]
	mineTree(myFPtree,myHeaderTab,minSup,set([]),myFreqList)
	return myFreqList
