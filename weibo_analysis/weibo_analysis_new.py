
# coding: utf-8

# In[1]:

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
logFile = open(os.path.join(resultFolder,'log.txt'),'w+')


# In[ ]:

#transData
logFile.write('test log\n')
start_t = time.time()
dataFiles = os.listdir(dataFolder)
csv
    # print dataFiles
for data_f in dataFiles:
    #readfile
    datafile = open(os.path.join(dataFolder,data_f),'r')
    csvfile = csv.writer(open(os.path.join(csvDataFolder,data_f+'.csv'),'w'))
    for line in datafile:
        org_vertic =line.split(',')[0].strip()
        dit_vertics = line.split(',')[1:]
        for dst_vertic in dit_vertics:
            csvfile.writerow([org_vertic,dst_vertic.strip()])
    print data_f

logFile.write('transfer file done! %s s\n' %str(time.time()-start_t))


# In[2]:

start_r = time.time()
sframeFiles = os.listdir(sframeDataFolder)
print sframeFiles


# In[ ]:

#sframe reading
edgesData = gl.SFrame()
for sframe_f in sframeFiles:
    print sframe_f
    edgesData = edgesData.append(gl.load_sframe(os.path.join(sframeDataFolder,sframe_f)))
                                     
edgesData.rename({'X1':'src','X2':'dst'})
print 'num_rows %d' %edgesData.num_rows()

# logFile.write('Sframe read data:%f s\n' %(time.time()-start_r))                                 
#save sframe data
edgesData.save(os.path.join(resultFolder,'weibo_sframe_data_400G'))


# In[2]:

edgesData= gl.load_sframe(os.path.join(resultFolder,'weibo_sframe_data_400G'))
print 'num_rows %d' %edgesData.num_rows()


# In[3]:

#generate Graph
start_g = time.time()
G = gl.SGraph()
G = G.add_edges(edges=edgesData,src_field='src',dst_field='dst')
print 'generate graph done! %f s\n' %(time.time()-start_g)

# logFile.write('generate graph done! %f s\n' %(time.time()-start_g))
# G_info =G.summary() 
# logFile.write('num_edges: %d, num_vertices: %d \n' %(G_info['num_edges'],G_info['num_vertices']))
# print G_info


# In[ ]:

#compute pagerank
pagerank_v = gl.pagerank.create(G)
print 'pagerank trainning time: %f s' %(pagerank_v['training_time'])


# In[ ]:

#add pagerank values for vertices
start_s = time.time()
G.vertices['pagerank'] = pagerank_v['graph'].vertices['pagerank']
logFile.write('save pagerank data: %f s\n' %(time.time()-start_s))
#save pagerank of nodes
node_pageValue = pagerank_v['pagerank']
node_pageValue.save(resultFolder+'/nodes_pagerank_400G.csv', format='csv')

logFile.close()
print 'done!'


# In[ ]:

print G.summary()
#save graph
G.save(os.path.join(resultFolder,'weibo_graph_400G'))
#load Graph
# G = gl.load_graph(os.path.join(resultFolder,'weibo_graph_400G'))


# In[ ]:

#save graph with vertices atrribute pagerank
G.save(os.path.join(resultFolder,'weibo_graph_400G'))


# 以上为400G微博用户关系网络特征统计,包括节点pagerank值,

# In[34]:

edges.rename({'X1':'src','X2':'dst'})
edges.save(os.path.join('/home/lt/datas/sframe','sFrame_edegs'))


# In[ ]:

edegs_n = gl.load_sframe(os.path.join(sframeDataFolder,'sFrame_1000'))
print edegs_n.num_rows()



# In[36]:

G = gl.SGraph()
G = G.add_edges(edegs_n, src_field = 'src',dst_field = 'dst')
G.summary()


# In[38]:

G.save(os.path.join('/home/lt/datas/out_data','graph_test'))


# In[ ]:




# In[33]:

edges = gl.SFrame.read_csv(os.path.join(csvDataFolder,'001008_0.csv'),header= False, delimiter=',',column_type_hints=int)
edegs_n


# In[42]:

from graphlab import SGraph, Vertex
g = SGraph().add_vertices([Vertex('cat', {'fluffy': 1}),
                               Vertex('dog', {'fluffy': 1, 'woof': 1}),
                               Vertex('hippo', {})])
g.vertices.save(os.path.join(resultFolder,'test_graph_vertices'),format='csv')


# In[59]:

#sframe reading
start_r = time.time()
# csvFiles = os.listdir(csvDataFolder)
csvFiles = ['000000_0.csv',
'000001_0.csv',
'000002_0.csv',
'000003_0.csv',
'000004_0.csv',
'000005_0.csv',
'000006_0.csv',
'000007_0.csv',
'000008_0.csv',
'000009_0.csv',
]
edgesData_t = gl.SFrame()
for csv_f in csvFiles:
    edgesData_t = edgesData_t.append(gl.SFrame.read_csv(os.path.join(csvDataFolder,csv_f),header=False,delimiter=',',column_type_hints=int))
edgesData_t.rename({'X1':'src','X2':'dst'})
print edgesData_t
print ('Sframe read data:%f s\n' %(time.time()-start_r))


# In[60]:

edgesData_t.save(os.path.join(resultFolder,'sframe_test_10'))


# In[49]:

for fileNum in xrange(0,50):
    csvfile = '%06d' %fileNum+'_0.csv'
print type(csvfile)


# In[63]:

totalFileNum = 1051

for num in xrange(100,totalFileNum,100):
    print num


# In[ ]:



