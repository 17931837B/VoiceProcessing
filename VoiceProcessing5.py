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


#累積距離正規化関数(斜め遷移2倍,r:整合窓)
def distanceGR(listA,listB,x,y,r):
  def distanceD(listA,listB,x,y,i,j):
    square_sum = 0
    for k in range(15):
      square_sum += (listA[x][i][k] - listB[y][j][k]) ** 2
    return math.sqrt(square_sum)
  
  I = len(listA[x])
  J = len(listB[y])
  g = 1000*np.ones((J,I),dtype=float)
  g[0][0] = distanceD(listA,listB,x,y,0,0) #初期条件
  for i in range(1,r): #境界条件1
    g[0][i] = g[0][i-1] + distanceD(listA,listB,x,y,i,0)
  for j in range(1,r): #境界条件2
    g[j][0] = g[j-1][0] + distanceD(listA,listB,x,y,0,j)
  for i in range(1,I):
    for j in range(1,J):
      if abs(J/I*i-j)<=r:
        g[j][i] = min(g[j][i-1]+distanceD(listA,listB,x,y,i,j), g[j-1][i-1]+2*distanceD(listA,listB,x,y,i,j), g[j-1][i]+distanceD(listA,listB,x,y,i,j))
  return g[J-1][I-1]/(I+J)


#整合窓(r=5)(011,012)
start = time.time()
r = 5
matchA1A2W = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceGR(FV11,FV12,x,y,r)
  if np.argmin(T) == x:
    matchA1A2W += 1

t = time.time() - start

print("単語認識率：",matchA1A2W,"%")
print("実行時間:",t,"秒")
#単語認識率： 98 %
#実行時間: 82.74494671821594 秒


#整合窓(r=4)(011,012)
start = time.time()
r = 4
matchA1A2W = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceGR(FV11,FV12,x,y,r)
  if np.argmin(T) == x:
    matchA1A2W += 1

t = time.time() - start

print("単語認識率：",matchA1A2W,"%")
print("実行時間:",t,"秒")
#単語認識率： 96 %
#実行時間: 68.36167025566101 秒


#整合窓(r=3)(011,012)
start = time.time()
r = 3
matchA1A2W = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceGR(FV11,FV12,x,y,r)
  if np.argmin(T) == x:
    matchA1A2W += 1

t = time.time() - start

print("単語認識率：",matchA1A2W,"%")
print("実行時間:",t,"秒")
#単語認識率： 95 %
#実行時間: 54.875572681427 秒


#整合窓(r=2)(011,012)
start = time.time()
r = 2
matchA1A2W = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceGR(FV11,FV12,x,y,r)
  if np.argmin(T) == x:
    matchA1A2W += 1

t = time.time() - start

print("単語認識率：",matchA1A2W,"%")
print("実行時間:",t,"秒")
#単語認識率： 95 %
#実行時間: 38.64344120025635 秒



#整合窓(r=1)(011,012)
start = time.time()
r = 1
matchA1A2W = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceGR(FV11,FV12,x,y,r)
  if np.argmin(T) == x:
    matchA1A2W += 1

t = time.time() - start

print("単語認識率：",matchA1A2W,"%")
print("実行時間:",t,"秒")
#単語認識率： 92 %
#実行時間: 22.75525403022766 秒


#整合窓(r=0)(011,012)
start = time.time()
r = 0
matchA1A2W = 0
for x in range(100):
  T = np.zeros(100)
  for y in range(100):
    T[y]= distanceGR(FV11,FV12,x,y,r)
  if np.argmin(T) == x:
    matchA1A2W += 1

t = time.time() - start

print("単語認識率：",matchA1A2W,"%")
print("実行時間:",t,"秒")
#単語認識率： 6 %
#実行時間: 6.15238881111145 秒
