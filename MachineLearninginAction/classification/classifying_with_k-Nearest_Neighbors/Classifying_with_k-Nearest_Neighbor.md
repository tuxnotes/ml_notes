# Classifying with k-Nearest Neighbors

# 1.1 Classifying with distance measurements

**k-Nearest Neighbors**

 - Pros: High accuracy, insensitive to outliers, no assumptions about data
 - Cons: Computationally expensive, requires a lot of memory
 - Works with: Numeric values, nominal values

**General approach to kNN**

1. Collect: Any method.
2. Prepare: Numeric values are needed for a distance calculation. A structured data format is best.
3. Analyze: Any method.
4. Train: Does not apply to the kNN algorithm.
5. Test: Calculate the error rate.
6. Use: This application needs to get some input data and output structured numeric values. Next, the application runs the kNN algorithm on this input data and determines which class the input data should belong to. The application then takes some action on the calculated class.

### 1.1.1 Prepare: importing data with Python

First, we'll create a Python module called kNN.py, where we'll place all the code used in this chapter.

```python
import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group, labels
```

这里有4条数据，每条数据有2个属性或特征。在分组矩阵中，每一行代表一条数据。标签向量给每个数据点一个标签。标签向量中元素的数量应该与分组数据中的行数相同。

### 1.1.2  Putting the kNN classification algorithm into action

本节将创建一个函数，用于在每条数据上运行kNN算法。需要记住，函数的目标是使用kNN算法将称为*inX*的数据进行分类，伪代码如下：

```
 For every point in our dataset:
	calculate the distance between inX and the current point
	sort the distances in increasing order
	take k items with lowest distances to inX
	find the majority class maong these items
	return the majority class as our prediction for the class of inX
```

Python代码如下：

```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

### 1.1.3 How to test a classifier

To test out a classifier, you start with some known data so you can hide the answer from the classifier and ask the classifier for its best guess.You can add up the number of times the classifier was wrong and divide it by the total number of tests you gave it. This will give you the error rate, which is a common measure to gauge how good a classifier is doing on a dataset. An error rate of 0 means you have a perfect classifier, and an error rate of 1.0 means the classifier is always wrong. You’ll see this in action with some solid data later.

We’re going to put k NN to use in real-world examples in the next two sections. First, we’ll look at improving the results from a dating site with k NN , and then we’ll look at an impressive handwriting
recognition example. We’ll employ testing in the handwriting recognition example to see if this algorithm is working.

## 1.2 示例:使用k-近邻算法改进约会网站的配对效果

示例：在约会网站上使用k-近邻算法

​	(1) 收集数据：提供文本文件；

​	(2) 准备数据：使用Python解析文本文件；

​	(3) 分析数据：使用Matplotlib画二维扩散图；

​	(4) 训练算法：此步骤不适用于k-近邻算法；

​	(5) 测试算法：使用海伦提供的部分数据作为测试样本

​		测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。

​	(6) 使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

### 1.2.1 准备数据：从文本文件中解析数据

海伦收集约会数据已经有一段时间了，这些数据存放在文本文件datingTestSet.txt中，每个样本数据占一行，总共1000行。海伦的样本主要包含以下3中特种：

 - 每年获得的飞行常客里程数
 - 玩视频游戏花费的时间百分比
 - 每周消费的冰淇淋公升数

将上述特征数据输入到分类器前，需要将数据格式转换为分类器接受的格式。因此在kNN.py中定义了一个新的函数*file2matrix* 。此函数以文件名作为参数输入，然后输出训练样本矩阵和类别标签向量。将下面的代码加到kNN.py

```python
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLableVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
```



