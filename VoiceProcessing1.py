import glob
import math
import numpy as np
import time

#city011の読み込み
file = glob.glob('./city_mcepdata/city011/*')
file.sort()

featureValue=[[] for _ in range(100)]
for i in range(100):
    f=open(file[i],"r")
    fileData=f.readlines()
    frameNum=int(fileData[2])
    
    for j in range(3,frameNum+3):
        frame =fileData[j].split(" ")
        del frame[-1]
        frameF=[float(s) for s in frame]
        featureValue[i].append(frameF)

f.close()
FV11=np.array(featureValue) #ファイル(100)*frame*dimension(15)


#city012の読み込み
file = glob.glob('./city_mcepdata/city012/*')
file.sort()

featureValue=[[] for _ in range(100)]
for i in range(100):
    f=open(file[i],"r")
    fileData=f.readlines()
    frameNum=int(fileData[2])
    
    for j in range(3,frameNum+3):
        frame =fileData[j].split(" ")
        del frame[-1]
        frameF=[float(s) for s in frame]
        featureValue[i].append(frameF)

f.close()
FV12=np.array(featureValue) #ファイル(100)*frame*dimension(15)



#city021の読み込み
file = glob.glob('./city_mcepdata/city021/*')
file.sort()

featureValue=[[] for _ in range(100)]
for i in range(100):
    f=open(file[i],"r")
    fileData=f.readlines()
    frameNum=int(fileData[2])
    
    for j in range(3,frameNum+3):
        frame =fileData[j].split(" ")
        del frame[-1]
        frameF=[float(s) for s in frame]
        featureValue[i].append(frameF)

f.close()
FV21=np.array(featureValue) #ファイル(100)*frame*dimension(15)


#city022の読み込み
file = glob.glob('./city_mcepdata/city022/*')
file.sort()

featureValue=[[] for _ in range(100)]
for i in range(100):
    f=open(file[i],"r")
    fileData=f.readlines()
    frameNum=int(fileData[2])
    
    for j in range(3,frameNum+3):
        frame =fileData[j].split(" ")
        del frame[-1]
        frameF=[float(s) for s in frame]
        featureValue[i].append(frameF)

f.close()
FV22=np.array(featureValue) #ファイル(100)*frame*dimension(15)

   
#累積距離正規化関数(斜め遷移2倍)
def distanceG(listA,listB,x,y):
  def distanceD(listA,listB,x,y,i,j):
    square_sum = 0
    for k in range(15):
      square_sum += (listA[x][i][k] - listB[y][j][k]) ** 2
    return math.sqrt(square_sum)
  I = len(listA[x])
  J = len(listB[y])
  g = np.zeros((J,I),dtype=float)
  g[0][0] = distanceD(listA,listB,x,y,0,0) #初期条件
  for i in range(1,I): #境界条件1
    g[0][i] = g[0][i-1] + distanceD(listA,listB,x,y,i,0)
  for j in range(1,J): #境界条件2
    g[j][0] = g[j-1][0] + distanceD(listA,listB,x,y,0,j)
  for i in range(1,I):
    for j in range(1,J):
      g[j][i] = min(g[j][i-1]+distanceD(listA,listB,x,y,i,j), g[j-1][i-1]+2*distanceD(listA,listB,x,y,i,j), g[j-1][i]+distanceD(listA,listB,x,y,i,j))
  return g[J-1][I-1]/(I+J) 


#同一話者１ 時間表示あり

start=time.time()

matchA1A2=0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceG(FV11,FV12,x,y)
  if np.argmin(T) == x:
    matchA1A2 += 1

t = time.time() - start

print("単語認識率：",matchA1A2,"%")
print("実行時間:",t,"秒")
#単語認識率： 99 %
#実行時間: 654.4030628204346 秒



#同一話者１ 
matchA1A2=0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceG(FV11,FV12,x,y)
  if np.argmin(T) == x:
    matchA1A2 += 1
print("単語認識率：",matchA1A2,"%")
#単語認識率： 99 %



#同一話者２
matchB1B2 = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceG(FV21,FV22,x,y)
  if np.argmin(T) == x:
    matchB1B2 += 1
print("単語認識率：",matchB1B2,"%")
#単語認識率： 99 %



#別話者(011,021)
matchA1B1 = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceG(FV11,FV21,x,y)
  if np.argmin(T) == x:
    matchA1B1 += 1
print("単語認識率：",matchA1B1,"%")
#単語認識率： 92 %



#別話者(011,022)
matchA1B2 = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceG(FV11,FV22,x,y)
  if np.argmin(T) == x:
    matchA1B2 += 1
print("単語認識率：",matchA1B2,"%")
#単語認識率： 88 %



#別話者(012,021)
matchA2B1 = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceG(FV12,FV21,x,y)
  if np.argmin(T) == x:
    matchA2B1 += 1
print("単語認識率：",matchA2B1,"%")
#単語認識率： 92 %



#別話者(012,022)
matchA2B2 = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceG(FV12,FV22,x,y)
  if np.argmin(T) == x:
    matchA2B2 += 1
print("単語認識率：",matchA2B2,"%")
#単語認識率： 88 %
