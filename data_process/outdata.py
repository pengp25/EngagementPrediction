import os

"""
This program is designed to Get the data processed by OPENFACE from DAiSEE Dataset
"""


#按不同数据集存储地址，对应修改filename和command命令地址

#python文件以斜杆/为辨识符
datadirfilename = "D:\\Aopenface\\DAiSEE\\DAiSEE\\DataSet"              #数据据根目录
datafilename = "D:/Aopenface/DAiSEE/DAiSEE/DataSet/Test"                #数据集Test根目录，若是数据集为Train，则对应修改数据集目录和输出目录
outputfilename = "D:/Aopenface/DAiSEE/DAiSEE/DataSet/output_Test"       #输出目录
#win命令以反斜杠\为辨识符
command = "D:\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe"           #exe文件地址
                   
                    ####43行需要把21改为当前数据集的人物数量####
                    ####    除此之外 以下代码地址无序更改  ####
  
#运行
for root, dirs, files in os.walk(datafilename):
    if(len(dirs)!=0):
        #print(len(dirs))
        if(len(dirs)!=21):
            #如果目录长度不等于21，即不为根目录
            for dir in dirs:
                input = root.replace('/','\\') +'\\' + dir +"\\" + dir + ".avi"               
                output = datadirfilename + "\\" + outputfilename.split('/')[-1] + "\\" + root.split('/')[-1].split('\\')[-1]
                print(output)
                finalcommand = command + " -f " + input + " -out_dir " + output
                os.system(finalcommand)

#删除不需要的文件（不是以.csv结尾的文件）
#最终会剩下 空文件夹 和 .csv文件 
for root, dirs, files in os.walk(outputfilename):
    for file in files:
        Array = map(file.endswith,".csv")
        if not (True in Array):
            targetfile = root.replace('\\','/') + "/" + file
            os.remove(targetfile)
#删除 空文件夹
for root, dirs, files in os.walk(outputfilename):
    if(len(dirs)!=0 and len(dirs)!=21):
    #0代表空文件，21代表数据集人物数量
        for dir in dirs:
            targetdir = root.replace('\\','/') + "/" + dir
            os.rmdir(targetdir)

