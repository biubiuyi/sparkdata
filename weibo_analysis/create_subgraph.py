
# coding: utf-8

# In[2]:

import graphlab as gl
import os

#data folder
resultDataFolder = '/home/lt/datas/out_data'
sFrameFolder = '/home/lt/datas/sFrame'


# In[ ]:

#load all edges
# edgesData_all = gl.load_sframe(os.path.join(resultDataFolder,'weibo_sframe_data_400G'))
# print 'num of edges:%d'%edgesData_all.num_rows()
# edgesData_all.head(5)

# #add edegs for subgraph
# subgraph_edges = gl.SFrame()

# #filter the neighbors of zhima usr
# zhima_neighbors = edgesData_all.filter_by(usr_id,'src')
# zhima_neighbors.head(5)

# subgraph_edges.append(zhima_neighbors)                                  


# In[7]:

#load zhima data
zhima_usr = gl.SFrame.read_csv('/home/lt/datas/zhima/zhima_score.csv',header=True, delimiter = ',')
usr_id  = zhima_usr['snwb']
print 'get the usr_id of zhima user'


# In[8]:

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
subgraph_edges.save(os.path.join(resultDataFolder,'subgraph_zhima'))


# In[1]:

print subgraph_edges.head(5)


# In[ ]:

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
    
#save subgraph
subgraph_edges.save(os.path.join(resultDataFolder,'subgraph_zhima'))


# In[ ]:

zhima_usr.rename({'snwb':'_id'})
#create graph
sub_G= gl.SGraph()
sub_G = sub_G.add_edges(edges = subgraph_edges,src_field = 'src',dst_field='dst')

#join label to vertices
sub_G.vertices.join(zhima_usr, on='_id', how='left')
sub_G.vertices.head(5)

