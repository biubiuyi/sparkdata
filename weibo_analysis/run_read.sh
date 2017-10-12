import graphlab as gl
import time
import csv
import os

#data Folder
dataFolder = '/home/lt/datas/t_base_weibo_user_fri_effect_user'
csvDataFolder = '/home/lt/datas/t_base_weibo_csv'
resultFolder = '/home/lt/datas/out_data'
sframeDataFolder = '/home/lt/datas/sFrame'
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)

def loadSframeData():
	edgesData= gl.load_sframe(os.path.join(resultFolder,'weibo_sframe_data_400G'))
    #create graph
    start_g = time.time()
    G = gl.SGraph()
    G = G.add_edges(edges=edgesData,src_field='src',dst_field='dst')
    print 'create graph done!'
    return G

def createSubgraph():
	G = loadSframeData()
	#load zhima uid
    zhima_usr = gl.SFrame.read_csv('/home/lt/datas/zhima/zhima_score.csv',header=True,delimiter = ',')
    usr_id = zhima_usr['snwb']

    #create subgraph of source vertices
    start = time.time()
    subgraph = G.get_neighborhood(usr_id, radius=1,full_subgraph=True)
    print 'create subgraph done: %f s\n' %(time.time()-start)

    #save subgraph
    subgraph.save(os.path.join(resultFolder,'subgraph_zhima_1'))
	print 'save subgraph done!'


if __name__ == '__main__':
	createSubgraph()