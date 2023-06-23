import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


#バックトラック
def distanceGP(list_a,list_b,x,y):
  def distanceD(list_a,list_b,x,y,i,j):
    square_sum = 0
    for k in range(15):
      square_sum += (list_a[x][i][k] - list_b[y][j][k]) ** 2
    return math.sqrt(square_sum)
  
  s = []
  t = []
  I = len(list_a[x])
  J = len(list_b[y])
  p = I-1
  q = J-1
  g = np.zeros((J,I),dtype=float)
  g[0][0] = distanceD(list_a,list_b,x,y,0,0) #初期条件
  for i in range(1,I): #境界条件1
    g[0][i] = g[0][i-1] + distanceD(list_a,list_b,x,y,i,0)
  for j in range(1,J): #境界条件2
    g[j][0] = g[j-1][0] + distanceD(list_a,list_b,x,y,0,j)
  for i in range(1,I):
    for j in range(1,J):
      g[j][i] = min(g[j][i-1]+distanceD(list_a,list_b,x,y,i,j), g[j-1][i-1]+2*distanceD(list_a,list_b,x,y,i,j), g[j-1][i]+distanceD(list_a,list_b,x,y,i,j))

  while p>0 or q>0:
    if g[q][p] ==  g[q][p-1]+distanceD(list_a,list_b,x,y,p,q):
      s.append(p-1)
      t.append(q)
      p -= 1
    elif g[q][p] == g[q-1][p-1]+2*distanceD(list_a,list_b,x,y,p,q):
      s.append(p-1)
      t.append(q-1)
      p -= 1
      q -= 1 
    elif g[q][p] == g[q-1][p]+distanceD(list_a,list_b,x,y,p,q):
      s.append(p)
      t.append(q-1)
      q -= 1
    
  return s,t

#(011,012)の描写
fig = plt.figure()
ax1 = Axes3D(fig)

for y in range(100):
  X,Y = distanceGP(FV11,FV12,0,y)
  Z = y
  
  ax1.plot(X,Y,Z,mew=0.5)

plt.title("Same speaker backtracking")
plt.show()
plt.close()

#(011,021)の描写
fig = plt.figure()
ax1 = Axes3D(fig)

for y in range(100):
  X,Y = distanceGP(FV11,FV21,0,y)
  Z = y
  
  ax1.plot(X,Y,Z,mew=0.5)

plt.title("Another Talker Backtrack")
plt.show()
plt.close()
