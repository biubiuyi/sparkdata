import graphlab as gl 
import os


def readCsv2Sframe(start_fileNum,stopfileNum):
	csvDataFolder = '/home/lt/datas/t_base_weibo_csv'
	outDataFolder = '/home/lt/datas/sFrame'
	csvFiles = os.listdir(csvDataFolder)

	#sframe reading
	edgesData_t = gl.SFrame()
	for fileNum in xrange(start_fileNum,stopfileNum):
		csv_f = '%06d' %fileNum + '_0.csv'
		if csv_f in csvFiles:
			edgesData_t = edgesData_t.append(gl.SFrame.read_csv(os.path.join(csvDataFolder,csv_f),header=False,delimiter=',',column_type_hints=int))
            print csv_f
    
    #save data
    edgesData_t.save(os.path.join(outDataFolder,'sFrame_%d'%stopfileNum))

    print 'save sframe data %d'%stopfileNum

readCsv2Sframe(0,100)