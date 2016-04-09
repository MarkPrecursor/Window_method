# -*- coding:utf-8 -*-  
import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import sympy;

##Parzen窗算法主程序,返回概率密度的估计值******************************************************//
def Parzen_Window(h):
    sum=0.0;
    for i in range(num):
        if(abs(a[i])<h/2 and abs(b[i])<h/2 and abs(c[i])<h/2):     sum+=1;
    sum=sum/float(num*math.pow(h,3));
    return sum;

##Knn算法主程序********************************************************************************//
def Knn_Window(n,D):
    h=D[n]+0.001;
    sum=0.0;
    for i in range(n):
        if(abs(a[i])<h/2 and abs(b[i])<h/2 and abs(c[i])<h/2):     sum+=1;
    sum=sum/float(n*math.pow(h,3));
    return sum;

#knn主程序，输入参数分别为数据，数据维度，样本个数，k值,自变量x********************************//
def Knn(Data,Dimension,n,k,x):
    if(Dimension==1):#Data为一维数据
        distance=[0]*n;
        for i in range(n):
            d=x-Data[i];                  #做差
            distance[i]=abs(d);
        distance.sort();                  #将差进行排序
        h=distance[k-1]+0.00000001;
        m_Sum=0;
        for i in range(n):
            m_Sum+=1/math.pow(6.28,0.5)*math.exp(-math.pow((x-Data[i]),2)/2);
        m_Sum/=n*math.pow(2*h,1);
        return m_Sum;
    elif(Dimension==2):#Data为二维数据
        distance=[0]*n;
        for i in range(n):
            d=x-Data[i,:];
            distance[i]=math.pow(d[0]**2+d[1]**2,0.5);
        distance.sort();
        h=distance[k-1]+0.00000001;#求得窗半径
        m_Sum=0;
        for i in range(n):
            u=x-Data[i,:];
            m_Sum+=1/math.pow(6.28,0.5)*math.exp(-(u[0]**2+u[1]**2)/2);##此处要求二维高斯窗
        m_Sum/=n*math.pow(2*h,2);
        return m_Sum;
    elif(Dimension==3):
        distance=[0]*n;
        for i in range(n):
            d=numpy.array(x-Data[i,:]);
            distance[i]=math.pow(d[0,0]**2+d[0,1]**2++d[0,2]**2,0.5);
        distance.sort();
        h=distance[k-1]+0.00000001;#求得窗半径
        m_Sum=0;
        for i in range(n):
            u=x-Data[i,:];
            m_Sum+=1/math.pow(6.28,0.5)*math.exp(-(u[0,0]**2+u[0,1]**2+u[0,2]**2)/2);##此处要求二维高斯窗
        m_Sum/=n*math.pow(2*h,3);
        return m_Sum;

##该程序对于样本库采用h从0到1变化，绘图显示原点附近的概率密度的变化****************************//
def Program_2():
    count=500;
    A=numpy.linspace(0,1,count);                             #用于储存h
    m_Data=numpy.zeros(count);                               #用于储存最后用于画图的点
    m_Mem=[0]*num;                                           #用于储存每个样本的运算值
    for n in range(1,count):    
        m_Data[n]=Parzen_Window(A[n]);
    lines=plt.plot(A,m_Data,'x',linestyle="-");
    plt.setp(lines, color='r', linewidth=2.0);
    plt.grid(True);
    plt.show(); 

##该程序在原点附近，使用刚好包含n个点的窗口，估计概率密度**************************************//
def Program_3():
    num=10000;
    #首先来计算每个点到原点的距离，对样本进行排序，这里每个样本以绝对值最大的元素为代表
    Distance=[0]*num;
    for i in range(num):    Distance[i]=max(abs(a[i]),abs(b[i]),abs(c[i]));
    Distance.sort();                                        #进行排序，精确地得到各个半径大小需要的值
    count=num/10;
    A=numpy.linspace(0,num,count);                          #用于储存h
    m_Data=numpy.zeros(count);                              #用于储存最后用于画图的点
    for n in range(1,count):
        m_Data[n]=Knn_Window(n*10,Distance);
    lines=plt.plot(A,m_Data,'x',linestyle="-");
    plt.setp(lines, color='b', linewidth=2.0);
    plt.grid(True);
    plt.show();

def Program_4():
    n=100;
    A=numpy.linspace(0,1,n);                          #用于储存h
    Data=[0]*n;
    X1=[1.36,1.41,1.22,2.46,0.68,2.51,0.60,0.64,0.85,0.66];
    for i in range(n):    Data[i]=Knn(X1,1,10,1,A[i]);
    plt.subplot(3,1,1);
    lines=plt.plot(A,Data,'x',linestyle="-");
    plt.setp(lines, color='r', linewidth=2.0);
    for i in range(n):    Data[i]=Knn(X1,1,10,3,A[i]);
    plt.subplot(3,1,2);
    lines=plt.plot(A,Data,'x',linestyle="-");
    plt.setp(lines, color='g', linewidth=2.0);
    for i in range(n):    Data[i]=Knn(X1,1,10,5,A[i]);
    plt.subplot(3,1,3);
    lines=plt.plot(A,Data,'x',linestyle="-");
    plt.setp(lines, color='b', linewidth=2.0);
    plt.grid(True);
    plt.show();

def Program_5():
    #mean=numpy.array([0,0]);
    #Var=numpy.matrix([[1,0],
    #                  [0,1]]);
    Data=[[0.011,1.03],
          [1.27,1.28],
          [0.13,3.12],
          [-0.21,1.23],
          [-2.18,1.39],
          [0.34,1.96],
          [-1.38,0.94],
          [-0.12,0.82],
          [-1.44,2.31],
          [0.26,1.94]];#numpy.random.multivariate_normal(mean,Var,1000);

    Data=numpy.array(Data);
    num=100;
    Z=numpy.zeros((num,num));
    r=4
    arrow=numpy.linspace(-r,r,num);

    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    for i in range(num):
        for j in range(num):
            a=numpy.array([arrow[i],arrow[j]]);
            Z[i,j]=Knn(Data,2,10,1,a);
    X=numpy.linspace(-r,r,num);
    Y=numpy.linspace(-r,r,num);
    X, Y = numpy.meshgrid(X,Y);
    ax.plot_surface(X,Y,Z,rstride = 1, cstride =1 , color='b');
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show();

def Program_6():
    W=[[0.28,1.31,-6.2],
       [0.07,0.58,-0.78],
       [1.54,2.01,-1.63],
       [-0.44,1.18,-4.32],
       [-0.81,0.21,5.73],
       [1.52,3.16,2.77],
       [2.20,2.42,-0.19],
       [0.91,1.94,6.21],
       [0.65,1.93,4.38],
       [-0.26,0.82,-0.96],
       [0.011,1.03,-0.21],
       [1.27,1.28,0.08],
       [0.13,3.12,0.16],
       [-0.21,1.23,-0.11],
       [-2.18,1.39,-0.19],
       [0.34,1.96,-0.16],
       [-1.38,0.94,0.45],
       [-0.12,0.82,0.17],
       [-1.44,2.31,0.14],
       [0.26,1.94,0.08],
       [1.36,2.17,0.14],
       [1.41,1.45,-0.38],
       [1.22,0.99,0.69],
       [2.46,2.19,1.31],
       [0.68,0.79,0.87],
       [2.51,3.22,1.35],
       [0.60,2.44,0.92],
       [0.64,0.13,0.97],
       [0.85,0.58,0.99],
       [0.66,0.51,0.88]];
    W=numpy.matrix(W)
    x1=numpy.array([-0.41,0.82,0.88]);
    x2=numpy.array([0.14,0.72,4.1]);
    x3=numpy.array([-0.81,0.61,-0.38]);
    Z1=Knn(W,3,30,5,x1);
    Z2=Knn(W,3,30,5,x2);
    Z3=Knn(W,3,30,5,x3);

    print('Data1',Z1)
    print('Data2',Z2)
    print('Data3',Z3)


##这里生成10000个均匀分布的立方体内的点
num=10000;
a=numpy.random.uniform(-0.5,0.5,num);
b=numpy.random.uniform(-0.5,0.5,num);
c=numpy.random.uniform(-0.5,0.5,num);
Program_2();
Program_3();
a=numpy.random.normal(0,1,num);
b=numpy.random.normal(0,1,num);
c=numpy.random.normal(0,1,num);
Program_2();
Program_3();
Program_4();
Program_5();
Program_6();
