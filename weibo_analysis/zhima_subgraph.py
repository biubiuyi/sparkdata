import graphlab as gl
import os

#data folder
resultDataFolder = '/home/lt/datas/out_data'
sFrameFolder = '/home/lt/datas/sFrame'

def loadZhima():
	#load zhima data
	zhima_usr = gl.SFrame.read_csv('/home/lt/datas/zhima/zhima_score.csv',header=True, delimiter = ',')
	print 'get the zhima user'
	return zhima_usr

def loadNeigh_1():
	print 'loadNeighbor 1'
	zhima_usr = loadZhima()
	usr_id  = zhima_usr['snwb']

	#add zhima 1_neighbor for subgraph 
	subgraph_edges = gl.SFrame()
	sframeFiles = os.listdir(sFrameFolder)
	for sf in sframeFiles:
	    edgesData = gl.load_sframe(os.path.join(sFrameFolder,sf))
	    edgesData.rename({'X1':'src','X2':'dst'})
	    zhima_neighbors = edgesData.filter_by(usr_id,'src')
	    subgraph_edges.append(zhima_neighbors) 
	    print sf
	    
	#save subgraph
	subgraph_edges.save(os.path.join(resultDataFolder,'subgraph_zhima_1'))

def loadNeigh_2():
	print 'loadNeighbor 2'
	zhima_usr = loadZhima()
	usr_id  = zhima_usr['snwb']

	#loadZhima_1
	subgraph_edges = gl.load_sframe(os.path.join(resultDataFolder,'subgraph_zhima_1'))

	#add zhima 2_neighbor for subgraph 
	neighbor_1 = subgraph_edges['dst']
	sub_vertices = usr_id.append(neighbor_1)

	# sframeFiles = os.listdir(sFrameFolder)
	for sf in sframeFiles:
	    edgesData = gl.load_sframe(os.path.join(sFrameFolder,sf))
	    edgesData.rename({'X1':'src','X2':'dst'})
	#     zhima_neighbors_2 = edgesData.filter_by(sub_vertices,'src')
	    zhima_neighbors_2 = edgesData[(edgesData['src'] in sub_vertices) & (edgesData['dst'] in sub_vertices)]
	    subgraph_edges.append(zhima_neighbors_2) 
	    print sf
	    
	print 'save subgraph'
	subgraph_edges.save(os.path.join(resultDataFolder,'subgraph_zhima_2'))

def createGraph():
	zhima_usr = loadZhima()
	zhima_usr.rename({'snwb':'_id'})

	subgraph_edges=gl.load_sframe(os.path.join(resultDataFolder,'subgraph_zhima_2'))
	#create graph
	sub_G= gl.SGraph()
	sub_G = sub_G.add_edges(edges = subgraph_edges,src_field = 'src',dst_field='dst')

	#join label to vertices
	sub_G.vertices.join(zhima_usr, on='_id', how='left')
	# sub_G.vertices.head(5)

	print 'save graph'
	sub_G.save(os.path.join(resultDataFolder,'subgraph_zhima'))

if __name__ == '__main__':
	loadNeigh_1()
	loadNeigh_2()
	createGraph()

