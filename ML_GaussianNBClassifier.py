'''
@func: Guassian NaiveBayes Classifier
@create: 2021-11-14
'''

def GaussianNBClassifierx():
    import streamlit as st

    # 高斯朴素贝叶斯，数据集为手写数字图片
    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline
    from sklearn.naive_bayes import GaussianNB
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X,y = digits.data,digits.target
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
    image_and_labels = list(zip(digits.images,digits.target))
    plt.figure(figsize=(4,3.5),dpi=100)
    for index,(image,label) in enumerate(image_and_labels[:12]):
        plt.subplot(3,4,index+1)
        plt.axis('off')
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title("Digit: %i" % label,fontsize=6)
    plt.show()


    '''图片数据一般使用像素点作为特征,
    由于图片的特殊性,相邻像素点间的数值(RGB三通道色)往往是接近的,
    故可以采用矩阵变换的方法压缩矩阵,得到相对较少的特征数
    数据总共包含1797张图片,每张图片的尺寸是8×8
    像素大小,共有十个分类(0-9),每个分类约180个样本.
    所有的图片数据保存在digits.image里,
    数据分析的时候需要转换成单一表格,即行为样本列为特征(类似的还有文档词矩阵),
    此案例中这个表格已经在digits.data里,可以通过digits.data.shape查看数据格式'''
    print("shape of raw image data: {0}".format(digits.images.shape))
    print("shape of data: {0}".format(digits.data.shape))

    #建模，探索建模结果
    gnb = GaussianNB().fit(Xtrain,Ytrain)
    #查看分数
    acc_score = gnb.score(Xtest,Ytest)
    print(acc_score)

    #查看预测结果
    Y_pred = gnb.predict(Xtest)
    Y_pred

    #查看预测的概率结果
    prob = gnb.predict_proba(Xtest) #每⼀列对应⼀个标签下的概率
    print(prob)

    #结果矩阵展示。矩阵中每一行的和都是一
    print(prob.shape) 


    #使⽤混淆矩阵查看贝叶斯的分类结果
    from sklearn.metrics import confusion_matrix as CM
    CM(Ytest,Y_pred)
    # 多分类状况下最佳的模型评估指标是混淆矩阵和整体的准确度


GaussianNBClassifierx()