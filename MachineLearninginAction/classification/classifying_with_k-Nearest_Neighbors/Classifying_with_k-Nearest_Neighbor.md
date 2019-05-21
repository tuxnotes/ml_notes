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

## 1.2 Example: improving matches from a dating site with kNN

