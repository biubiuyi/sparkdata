#author = 'lt'

import graphlab as gl
import time
import csv
import os

#data Folder
resultFolder = '/home/lt/out_data'
sframeDataFolder = '/home/lt/weibo_sframe'
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)

def loadSubData():
	sframeFiles = os.listdir('/home/lt/sframe')
	edgesData = gl.sFrame()
	for sf in sframeFiles:
		edgesData.append(gl.load_sframe('/home/lt/sframe',sf))
	edgesData.rename({'X1':'src','X2':'dst'})

	#create graph
	G = gl.SGraph()
	G = G.add_edges(edges = edgesData, src_field ='src',dst_field = 'dst')
	pritn 'create graph done!'
	return G


def loadData():
	edgesData = gl.load_sframe(sframeDataFolder)
	print 'num_rows:%d ' %edgesData.num_rows()

	#create graph
	G = gl.SGraph()
	G = G.add_edges(edges = edgesData, src_field ='src',dst_field = 'dst')
	pritn 'create graph done!'
	return G

def saveGraph(G):
	#save graph
	G.save(os.path.join(resultFolder,'weibo_graph_400G'))

def computePagerank():
	G = loadSubData()

	#pagerank
	pagerank_v = gl.pagerank.create(G)
	print 'pagerank training time:%s s' %str(pagerank_v['training_time'])

	#save pagerank
	node_pagerank = pagerank_v['pagerank']
	node_pagerank.save(os.path.join(resultFolder,'node_pagerank_400G.csv'),format = 'csv')

	#save graph
	G.vertices['pagerank'] = pagerank_v['graph'].vertices['pagerank']
    saveGraph(G)

def createSubgraph():
	G = loadSubData()
	#load zhima uid
    zhima_usr = gl.SFrame.read_csv('/home/lt/zhima/zhima_score.csv',header=True,delimiter = ',')
    usr_id = zhima_usr['snwb']

    #create subgraph of source vertices
    start = time.time()
    subgraph = G.get_neighborhood(usr_id, radius=1,full_subgraph=True)
    print 'create subgraph done: %f s\n' %(time.time()-start)

    #save subgraph
    subgraph.save(os.path.join(resultFolder,'subgraph_zhima_1'))
	print 'save subgraph done!'

# def labelPropagation():



if __name__ == '__main__':
	print 'compute pagerank'
	computePagerank()
	# print 'create subgraph of zhima'
	# createSubgraph()

