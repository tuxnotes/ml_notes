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

