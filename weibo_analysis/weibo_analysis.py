
# coding: utf-8

# In[3]:

import graphlab as gl
import csv
import os

#data file
dataFolder = '/home/lt/datas/weibo'
resultFolder = '/home/lt/datas/out_data'
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)


# In[8]:

#readfile
datafile = open(os.path.join(dataFolder,'t_base_weibo_user_fri_part000_9G'),'r')
csvfile = csv.writer(open(os.path.join(dataFolder,'t_base_weibo_user_fri_part000_9G.csv'),'w'))
for line in datafile:
    org_vertic =line.split('\001')[0].strip()
    dit_vertics = line.split('\001')[1].split(',')
    for dst_vertic in dit_vertics:
        csvfile.writerow([int(org_vertic),int(dst_vertic.strip())])

print 'transfile done!'


# In[11]:

#load data
edegsData = gl.SFrame.read_csv(os.path.join(dataFolder,'t_base_weibo_user_fri_part000_9G.csv'),header=False)
edegsData.rename({'X1':'src','X2':'dst'})


# In[12]:

#create graph
G = gl.SGraph()
G = G.add_edges(edges=edegsData,src_field='src',dst_field='dst')
print G.summary()


# In[17]:

#pagerank
pagerank_v = gl.pagerank.create(G,verbose=False)
print pagerank_v['training_time']


# In[18]:

node_pageValue = pagerank_v['pagerank']
node_pageValue.save(resultFolder+'/nodes_pagerank.csv', format='csv')


# In[9]:

sf = gl.SFrame()
sf2 = gl.SFrame([3,5,7,8,6,9])
sf = sf2.append(sf)
sf


# In[ ]:



