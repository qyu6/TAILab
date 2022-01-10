'''
@function:显示每个模块的原始代码
@create:2021.10.20
'''
import streamlit as st






# [4]机器学习模型API - 通用模块 - 混淆矩阵
def ConfusionMatrixcode():
    st.code('''
# 混淆矩阵
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Show confusion matrix
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(confusion_mat)

# Print classification report
from sklearn.metrics import classification_report
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print(classification_report(y_true, y_pred, target_names=target_names))
    ''')



# [4]机器学习模型API - 分类模型 - 朴素贝叶斯分类器
def NaiveBayesClassifiercode():
    st.code('''

# 用贝叶斯定理来进行建模的监督学习分类器
# cross_validation被替换为model_selection
# from sklearn.cross_validation import train_test_split 替换为：from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


# visualize classification results:
def plot_classifier(classifier, X, y):
    # define ranges to plot the figure 
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot 
    fig,ax = plt.subplots()

    # choose a color scheme you can find all the options based on mesh_output
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    ax.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.cividis)

    # Overlay the training points on the plot 
    ax.scatter(X[:, 0], X[:, 1], c=y, s=60, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.show()
    # st.pyplot(fig)



input_file = 'data/data_multivar(2).txt'

X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1]) 

X = np.array(X)
y = np.array(y)

# plt.scatter(X[:,0],y,c='r',marker='o')
# plt.show()


classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)


# compute accuracy of the classifier
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("(Before train/test split) Accuracy of the classifier =", round(accuracy, 2), "%")

# plot_classifier(classifier_gaussiannb, X, y)

###############################################
# Train test split
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb_new.predict(X_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("(After train/test split) Accuracy of the classifier =", round(accuracy, 2), "%")

plot_classifier(classifier_gaussiannb_new, X_test, y_test)

###############################################
# Cross validation and scoring functions

print('(After apply cross-validation ↓)')

num_validations = 5
accuracy = model_selection.cross_val_score(classifier_gaussiannb, 
        X, y, scoring='accuracy', cv=num_validations)
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

f1 = model_selection.cross_val_score(classifier_gaussiannb, 
        X, y, scoring='f1_weighted', cv=num_validations)
print("F1: " + str(round(100*f1.mean(), 2)) + "%")

precision = model_selection.cross_val_score(classifier_gaussiannb, 
        X, y, scoring='precision_weighted', cv=num_validations)
print("Precision: " + str(round(100*precision.mean(), 2)) + "%")

recall = model_selection.cross_val_score(classifier_gaussiannb, 
        X, y, scoring='recall_weighted', cv=num_validations)
print("Recall: " + str(round(100*recall.mean(), 2)) + "%")

    ''')


# [4]机器学习模型API - 分类模型 - 逻辑回归分类器
def LogisticsRegressionClassifiercode():
    st.code('''
# 逻辑回归是一种分类方法，给定一组数据点，建立一个可以在类之间绘制线性边界的模型
import numpy as np
from sklearn import linear_model 
import matplotlib.pyplot as plt

def plot_classifier(classifier, X, y):
    # define ranges to plot the figure 
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot 
    plt.figure()

    # choose a color scheme you can find all the options 
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.show()

if __name__=='__main__':
    # input data
    X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # initialize the logistic regression classifier
    # solver-用于设置求解系统方程的算法类型；C-表示正则化强度，数值越小，正则化强度越高
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

    # train the classifier
    classifier.fit(X, y)

    # draw datapoints and boundaries
    plot_classifier(classifier, X, y)

    
    ''')






# [4]机器学习模型API - 回归模型 - 随机森林回归器
def RandomForestRegressorcode():
    st.code('''
    
def RandomForestRegressorx1():

    # 随机森林是一个决策树集合，它基本上就是用一组由数据集的若干子集构建的决策树构成。
    # 再用决策树平均值改善整体学习效果
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn import preprocessing
    from sklearn.metrics import mean_squared_error, explained_variance_score
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt

    def plot_feature_importances(feature_importances, title, feature_names):
        # Normalize the importance values 
        feature_importances = 100.0 * (feature_importances / max(feature_importances))

        # Sort the values and flip them
        index_sorted = np.flipud(np.argsort(feature_importances))

        # Arrange the X ticks
        pos = np.arange(index_sorted.shape[0]) + 0.5

        # Plot the bar graph
        plt.figure()
        plt.bar(pos, feature_importances[index_sorted], align='center',color='blue')
        plt.xticks(pos, feature_names[index_sorted],rotation=30)
        plt.ylabel('Relative Importance')
        plt.grid(alpha=0.3)
        plt.title(title)
        plt.show()



    # Load the dataset from the input file
    # X, y, feature_names = load_dataset('work/bike_day.csv')
    df = pd.read_csv('data/bike_day.csv')
    X = np.array(df.iloc[0:len(df),2:13])
    y = np.array(df.iloc[0:len(df),-1])
    feature_names = np.array(list(df.iloc[:,2:13]))


    X, y = shuffle(X, y, random_state=7) 

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.9 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit Random Forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
    rf_regressor.fit(X_train, y_train)

    # Evaluate performance of Random Forest regressor
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred) 
    print("\n#### Random Forest regressor performance ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    # Plot relative feature importances 
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest Regressor', feature_names)


def RandomForestRegressorx2():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn import preprocessing
    from sklearn.metrics import mean_squared_error, explained_variance_score
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt

    def plot_feature_importances(feature_importances, title, feature_names):
        # Normalize the importance values 
        feature_importances = 100.0 * (feature_importances / max(feature_importances))

        # Sort the values and flip them
        index_sorted = np.flipud(np.argsort(feature_importances))

        # Arrange the X ticks
        pos = np.arange(index_sorted.shape[0]) + 0.5

        # Plot the bar graph
        plt.figure()
        plt.bar(pos, feature_importances[index_sorted], align='center',color='blue')
        plt.xticks(pos, feature_names[index_sorted],rotation=30)
        plt.ylabel('Relative Importance')
        plt.grid(alpha=0.3)
        plt.title(title)
        plt.show()


    # Load the dataset from the input file
    # X, y, feature_names = load_dataset('work/bike_day.csv')
    df = pd.read_csv('data/bike_hour.csv')
    X = np.array(df.iloc[0:len(df),2:14])
    y = np.array(df.iloc[0:len(df),-1])
    feature_names = np.array(list(df.iloc[:,2:14]))


    X, y = shuffle(X, y, random_state=7) 

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.9 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit Random Forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
    rf_regressor.fit(X_train, y_train)

    # Evaluate performance of Random Forest regressor
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred) 
    print("\n#### Random Forest regressor performance ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    # Plot relative feature importances 
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest Regressor', feature_names)
    ''')



# [5]训练开发工具 - 文件夹树结构可视化工具
def FolderTreeStructurecode(): 
    st.code('''
# 显示目标路径下(文件夹路径)的树形结构可视化
from pathlib import Path

class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


# test
target_path = 'C:/Users/Tony/Downloads/2021-11-Taskforce/Python-Machine-Learning-Cookbook-master'
paths = DisplayablePath.make_tree(Path(target_path))
for path in paths:
    print(path.displayable())
    ''')


# [4]机器学习模型API - 回归模型 - AdaBoost决策树回归器
def AdaboostDTRegressorcode():
    st.code('''
# 基于AdaBoost的决策树回归器. 
# Adaboost = Adaptive Boosting是指自适应增强算法，是一种利用其它系统增强模型准确性的技术。
# 这种技术基于是将不同的算法进行组合，用加权汇总的方式获得最终结果，被称为弱学习器(weak learners)
# AdaBoost算法在每个阶段获取的信息都会反馈到模型中，这样学习器就可以在后一阶段训练难以分类的样本
# 首先使用AdaBoost算法对数据集进行回归拟合，再计算误差，然后根据误差评估结果，用同样数据集重新拟合。可看做是回归器的调优过程，直到达到预期的准确性
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values 
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center',color='blue')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.grid(alpha=0.2)
    plt.title(title)
    plt.show()

if __name__=='__main__':
    # Load housing data
    housing_data = datasets.load_boston() 

    # Shuffle the data
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit decision tree regression model
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(X_train, y_train)

    # Fit decision tree regression model with AdaBoost
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ab_regressor.fit(X_train, y_train)

    # Evaluate performance of Decision Tree regressor
    y_pred_dt = dt_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_dt)
    evs = explained_variance_score(y_test, y_pred_dt) 
    print("\n#### Decision Tree performance ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    # Evaluate performance of AdaBoost
    y_pred_ab = ab_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_ab)
    evs = explained_variance_score(y_test, y_pred_ab) 
    print("\n#### AdaBoost performance ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    # Plot relative feature importances 
    plot_feature_importances(dt_regressor.feature_importances_, 
            'Decision Tree Regressor', housing_data.feature_names)
    plot_feature_importances(ab_regressor.feature_importances_, 
            'AdaBoost Regressor', housing_data.feature_names)
    ''')



# [3]机器学习模型API - 回归模型 - 多项式回归器
def polynomialregcode():
    st.code('''
# 多项式回归
import sys
import numpy as np
from sklearn import linear_model


# filename = sys.argv[1]
X = []
y = []
with open('data/data_multivar.txt', 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        y.append(yt)

# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
#X_train = np.array(X[:num_training]).reshape((num_training,1))
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])

# Test data
#X_test = np.array(X[num_training:]).reshape((num_test,1))
X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

from sklearn.preprocessing import PolynomialFeatures
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
print(X_train)
datapoint = [[0.39,2.78,7.11],[0.39,2.78,7.11]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nLinear regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))


# (SGD=随机梯度下降)Stochastic Gradient Descent regressor:随机梯度下降.是不将所有样本都进行计算后再更新参数，
# 而是选取一个样本，计算后就更新参数)
# (BGD=批量梯度下降)将所有样本数据都计算完成后再更新参数的方法叫做批量梯度下降
sgd_regressor = linear_model.SGDRegressor(loss='huber', n_iter_no_change=50)
sgd_regressor.fit(X_train, y_train)
print("\nSGD regressor:\n", sgd_regressor.predict(datapoint))
    ''')


# [3]机器学习模型API - 回归模型 - 岭回归器
def ridgeregcode():
    st.code('''
# 岭回归|通过引入正则化项的系数作为阈值来消除异常值的影响(普通最小二乘法会将每个点都考虑进去，进而带来误差)
import sys
import numpy as np

# filename = sys.argv[1]
X = []
y = []
with open('data/data_multivar.txt', 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        y.append(yt)

# alpha控制岭回归器的复杂程度。Alpha=0，相当于普通最小二乘法，如果希望对异常值不那么敏感，则增加alpha的值
# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
#X_train = np.array(X[:num_training]).reshape((num_training,1))
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])

# Test data
#X_test = np.array(X[num_training:]).reshape((num_test,1))
X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)
ridge_regressor.fit(X_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(X_test)
y_test_pred_ridge = ridge_regressor.predict(X_test)

# Measure performance
import sklearn.metrics as sm

print("LINEAR:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

print("\nRIDGE:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2)) 
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2))
    ''')



# [3]机器学习模型API - 回归模型 - 线性回归器
def linregcode():
    st.code('''
import sys
import numpy as np

X = [1,2,3,4,6,7,8,9,11,13,14,15,15,16,17,18,19,20]
y = [12,23,34,45,56,67,78,80,100,30,114,124,131,142,151,167,50,40]

num_training = int(0.8*len(X))
num_test = len(X) - num_training

X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

from sklearn import linear_model
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train,y_train)

import matplotlib.pyplot as plt
y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.grid(alpha=0.4)
plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,y_train_pred,color='black',linewidth = 1)
plt.title('Training data')
plt.show()

y_test_pred = linear_regressor.predict(X_test)
plt.figure()
plt.grid(alpha=0.4)
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_test_pred,color='black',linewidth = 1)
plt.title('Test data')
plt.show()

# 评价回归拟合器误差，几个重要的指标(metrics)
# 平均绝对误差-MAE:mean absolute error
# 均方误差-MES:mean squared error,误差平方均值
# 中位数绝对误差-median absolute error,用中位数可以消除异常值outlier的干扰
# 解释方差分-EVS:explained variance score,对数据集波动的解释能力
# R^2得分-拟合优度
# [通常做法]:保证均方误差最低，解释方差分最高
import sklearn.metrics as sm
print("Mean absolute error=",round(sm.mean_absolute_error(y_test,y_test_pred),2))
print("Mean squared error=",round(sm.mean_squared_error(y_test,y_test_pred),2))
print("Median absolute error=",round(sm.median_absolute_error(y_test,y_test_pred),2))
print("Explained variance score=",round(sm.explained_variance_score(y_test,y_test_pred),2))
print("R^2 score=",round(sm.r2_score(y_test,y_test_pred),2))



# # 以下内容测试没有通过。一种保存模型数据的方法
# # 保存模型数据:
# import cPickle as pickle
# output_model_file = 'saved_model.pkl'
# with open(output_model_file,'w') as f:
#     pickle.dump(linear_regressor,f)

# # 加载并使用数据：
# with open(output_model_file,'r') as f:
#     model_linregr = pickle.load(f)

# y_test_pred_new = model_linregr.predict(X_test)
# print('New mean absolute error=',round(sm.mean_absolute_error(y_test,y_test_pred_new),2))
    ''')


# [5]-训练开发工具 - OCR-Optical Character Recognition，光学字符识别
def ocrx1code():
    st.code('''
def ocr(img_path,language):
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image

    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite("removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(img_path, img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(img_path),lang=language)
    print(result)
    # Remove template file

    return result

ocr('pics/ocr_test.png','chi_sim')
    ''')


# [3]机器学习模型API - 数据预处理API
def preprocessingcode():
    st.code(
'''
import numpy as np
from sklearn import preprocessing

print('source-dataset:')
data = np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
print(data)

# 标准化处理
data_standardized = preprocessing.scale(data)
print('ScaledData',data_standardized)
print("ScaledMean=",data_standardized.mean(axis=0))
print("Scaled Std deviation=",data_standardized.std(axis=0))

# 范围缩放
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaler = data_scaler.fit_transform(data)
print("MinMaxScaled data=",data_scaler)

# 归一化
data_normalized = preprocessing.normalize(data,norm='l1')
print("L1 Normalized data=",data_normalized)

# 二值化
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("Binarized_data=",data_binarized)

# 独热编码
print('---')
encoder = preprocessing.OneHotEncoder()
data1 = np.array([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
print(data1)
encoder.fit([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
encoder_vector = encoder.transform([[2,3,5,3]]).toarray()
print('[2,3,5,3]')
print('OneHot Encoded Vector=',encoder_vector)

# 标记编码: string feature → number
print('---')
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classess = ['audi','ford','audi','toyota','ford','bmw']
print(input_classess)
label_encoder.fit(input_classess)
print('Class-mapping:')
for i,item in enumerate(label_encoder.classes_):
    print(item,'→',i)

print('→Use the label:')
labels = ['toyota','ford','audi']
encoded_labels=label_encoder.transform(labels)
print('Labels=',labels)
print('Encoded labels=',list(encoded_labels))

print('→Back search the label:')
encoded_labels = [2,1,0,3,1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print('Encoded labels=',encoded_labels)
print('Decoded labels=',decoded_labels)
'''
)
