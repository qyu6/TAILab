'''
author:T
create by:2021.10
function:TAI-Lab = Tony's AI Lab for solution research.
refer:https://docs.streamlit.io/library/api-reference
'''

# latest update:
releasemark = 'Dec-06,2021'


# env.requirements
from pathlib import WindowsPath
import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
import datetime
from PIL import Image
from CodingTips import *
# from ocr import ocr
# from embedded_pdf import st_display_pdf
from SCodingShow import *



# TAI-Lab Main Content
def main():    
    st.sidebar.header('©TAI-Lab')
    # st.sidebar.markdown('https://github.com/tqthooo2021/TAI-Lab')
    st.sidebar.markdown('📧<tqthooo2021@163.com>')
    st.sidebar.write('📅Latest release date: '+str(releasemark))
    


    # 主模块
    labfunc = st.sidebar.selectbox('lab-funcs:',
    [
        '[1]-在线python编程环境',
        '[2]-统计学模型API',
        '[3]-机器学习模型API',
        '[4]-深度学习模型API',
        '[5]-训练开发工具☆',
        '[6]-其他',
    ])



    if labfunc == '[1]-在线python编程环境':
        st.markdown('在线python编程环境(模型API可在此运行)') 
        source1 = st.sidebar.file_uploader('(Data-1)')
        source2 = st.sidebar.file_uploader('(Data-2)')
        source3 = st.sidebar.file_uploader('(Data-3)')

        # dataset-1
        if source1 is None:
            nothing = 'do nothing'
        else:
            df1 = pd.read_csv(source1)

        # dataset-2
        if source2 is None:
            nothing = 'do nothing'
        else:
            df2 = pd.read_csv(source2)

        # dataset-3
        if source3 is None:
            nothing = 'do nothing'
        else:
            df3 = pd.read_csv(source3)

        # code execution
        codex = st.text_area('code here:')
        if st.button('run'):
            with st.spinner('please wait...'):
                if len(codex)>0:                    
                    output = exec(codex)
                else:
                    st.markdown('Input code then click run.')
        else:
            nothing = 'n/a'
            


    elif labfunc == '[2]-统计学模型API':
        st.markdown('coming soon..')



    elif labfunc == '[3]-机器学习模型API':
        mlmodel = st.sidebar.selectbox('ml-models:',
        [
            '通用模块',        
            '回归模型',
            '分类模型',
            '聚类模型',
        ]
        )
        
        if mlmodel == '通用模块':
            cmmodel = st.sidebar.selectbox('co-models:',
            [
                '数据预处理',
                '混淆矩阵',
            ]
            )

            if cmmodel == '数据预处理':
                from ML_DataPreProcessing import preprocessingx
                st.subheader('数据预处理API')
                st.warning('预处理方法API:\n* preprocessing.scale(data) - 标准化(均值移除，消除偏差)\n'
                '* preprocessing.MinMaxScaler(feature_range=(0,1)) - 范围缩放(每个特征点分布在0~1之间)\n'
                '* preprocessing.normalize(data,norm="l1") - 归一化(每个特征缩放到相同的数据范围，特征向量调整为L1范数)\n'
                '* preprocessing.Binarlizer(threshold=1.4).transform(data) - 二值化(特征向量转化为布尔值)\n'
                '* preprocessing.OneHotEncoder() - 独热编码(将稀疏散乱数据进行编码，将每个特征与非重复总数特征相对应，是一种收紧特征向量的工具)\n'
                '* preprocessing.LabelEncoder() - 标记编码(将类别型变量转化为数值型)\n'
                '---\n'
                '* from ML_DataPreProcessing import preprocessingx\n'
                '* exec(str(preprocessingx()))\n'
                )

                with st.expander('show code'):
                    # from SCodingShow import preprocessingcode
                    exec(str(preprocessingcode()))

                st.write('----') 
                with st.spinner('please wait...'):
                    exec(str(preprocessingx()))     

            elif cmmodel == '混淆矩阵':
                from ML_ConfusionMatrix import ConfusionMatrixx
                st.subheader('混淆矩阵|Confusion Matrix')
                st.warning('混淆矩阵API:\n'
                '* from sklearn.metrics import confusion_matrix\n'
                '* from sklearn.metrics import classification_report\n'
                '---\n'
                '* from ML_ConfusionMatrix import ConfusionMatrixx\n'
                '* exec(str(ConfusionMatrixx()))\n' 
                )
                with st.expander('show code'):
                    exec(str(ConfusionMatrixcode()))
                st.write('---')
                with st.spinner('please wait...'):
                    exec(str(ConfusionMatrixx()))


            else:
                nothing = 'do nothing'


        elif mlmodel == '回归模型':
            reg_model = st.sidebar.selectbox('regression-model:',
            [
                '线性回归器',
                '岭回归器',
                '多项式回归器',
                'AdaBoost决策树回归器',
                '随机森林回归器',
            ]
            )
            
    
            if reg_model == '线性回归器':
                from ML_LinearRegression import linearregx
                st.subheader('线性回归器|Linear Regression')
                st.warning('线性回归器API:\n'
                '* from sklearn import linear_model\n'
                '* linear_model.LinearRegression()\n'
                '* linear_regressor.fit(X_train,y_train)\n'
                '* import sklearn.metrics as sm\n'
                '* sm.mean_absolute_error(y_test,y_test_pred)\n'
                '* sm.mean_squared_error(y_test,y_test_pred)\n'
                '* sm.median_absolute_error(y_test,y_test_pred)\n'
                '* sm.explained_variance_score(y_test,y_test_pred)\n'
                '* sm.r2_score(y_test,y_test_pred)\n'
                '---\n'
                '* from ML_LinearRegression import linearregx\n'
                '* exec(str(linearregx()))\n'
                )

                with st.expander('show code'):            
                    exec(str(linregcode()))

                st.write('----') 
                with st.spinner('Please wait...'):
                    exec(str(linearregx()))     


            elif reg_model == '岭回归器':
                from ML_RidgeRegression import RidgeRegx
                st.subheader('岭回归器|Ridge Regression(L2正则项)')
                st.warning('岭回归器API:\n'
                '* from sklearn import linear_model\n'
                '* ridge_regressor = linear_model.Ridge(alpha=0.01,fit_intercept=True,max_iter=10000)\n'
                '* ridge_regressor.fit(X_train,y_train)\n'
                '* y_test_pred_ridge = ridge_regressor.predict(X_test)\n'
                '---\n'
                '* from ML_RidgeRegression import RidgeRegx\n'
                '* exec(str(RidgeRegx()))\n'
                )

                with st.expander('show code'): 
                    exec(str(ridgeregcode()))

                st.write('----') 
                with st.spinner('Please wait...'):
                    exec(str(RidgeRegx())) 


            elif reg_model == '多项式回归器':
                from ML_PolynomialFeatures import PolyRegx
                st.subheader('多项式回归器|Polynomial Regression')
                st.warning('多项式回归器API:\n'
                '* from sklearn import linear_model\n'
                '* from sklearn.preprocessing import PolynomialFeatures\n'
                '* polynomial = PolynomialFeatures(degree=10)\n'
                '* X_train_transformed = polynomial.fit_transform(X_train)\n'
                '* sgd_regressor = linear_model.SGDRegressor(loss="huber", n_iter_no_change=50)\n'
                '* sgd_regressor.fit(X_train, y_train) | SGD:随机梯度下降，BGD:批量梯度下降\n'
                '---\n'
                '* from ML_PolynomialFeatures import PolyRegx\n'
                '* exec(str(PolyRegx()))\n'
                )
                with st.expander('show code'):
                    exec(str(polynomialregcode()))
                st.write('----')
                with st.spinner('Please wait...'):
                    exec(str(PolyRegx()))


            elif reg_model == 'AdaBoost决策树回归器':
                from ML_AdaboostDTRegressor import AdaboostDTRegressorx
                st.subheader('自适应增强决策树回归器|AdaBoost Decision Tree Regressor')
                st.warning('AdaBoost决策树回归器API:\n'
                '* from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor\n'
                '* from sklearn.tree import DecisionTreeRegressor\n'
                '* from sklearn import datasets\n'
                '* from sklearn.metrics import mean_squared_error, explained_variance_score\n'
                '* from sklearn.utils import shuffle\n'
                '* dt_regressor = DecisionTreeRegressor(max_depth=4)\n'
                '* ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)\n'
                '---\n'
                '* from ML_AdaboostDTRegressor import AdaboostDTRegressorx\n'
                '* exec(str(AdaboostDTRegressorx()))\n'
                )
                with st.expander('show code'):
                    exec(str(AdaboostDTRegressorcode()))
                st.write('----')
                with st.spinner("Please wait..."):
                    exec(str(AdaboostDTRegressorx()))

            
            elif reg_model == '随机森林回归器':
                from ML_RandomForestRegressor import RandomForestRegressorx1
                from ML_RandomForestRegressor import RandomForestRegressorx2               
                st.subheader('随机森林回归器|RandomForest Regressor')
                st.warning('随机森林回归器API:\n'
                '* from sklearn.ensemble import RandomForestRegressor\n'
                '* from sklearn import preprocessing\n'
                '* from sklearn.metrics import mean_squared_error, explained_variance_score\n'
                '* from sklearn.utils import shuffle\n'
                '* import matplotlib.pyplot as plt\n'
                '---\n'
                '* from ML_RandomForestRegressor import RandomForestRegressorx1\n'
                '* from ML_RandomForestRegressor import RandomForestRegressorx2\n'
                '* exec(str(RandomForestRegressorx1()))\n'
                '* exec(str(RandomForestRegressorx2()))\n'
                )
                with st.expander('show code'):
                    exec(str(RandomForestRegressorcode()))
                st.write('----')
                with st.spinner('Please wait...'):
                    exec(str(RandomForestRegressorx1()))
                    exec(str(RandomForestRegressorx2()))





            else:
                nothing = 'do nothing' 


        elif mlmodel == '分类模型':
            cal_model = st.sidebar.selectbox('classification model:',
            [
                '逻辑回归分类器',
                '朴素贝叶斯分类器',
                '高斯朴素贝叶斯分类器',
            ]
            )



            if cal_model == '逻辑回归分类器':
                from ML_LogisticRegressionClassifier import LogisticRegressionClassifierx
                st.subheader('逻辑回归分类器|Logistics Regression Classifier')
                st.warning('逻辑回归分类器API:\n'         
                '* from sklearn import linear_model\n'
                '* classifier = linear_model.LogisticRegression(solver="liblinear", C=100)\n'
                '* classifier.fit(X, y)\n'
                '---\n'
                '* from ML_LogisticRegr     essionClassifier import LogisticRegressionClassifierx\n'
                '* exec(str(LogisticRegressionClassifierx()))'
                )
                with st.expander('show code'):
                    exec(str(LogisticsRegressionClassifiercode()))
                st.write('----')
                with st.spinner('please wait...'): 
                    exec(str(LogisticRegressionClassifierx()))


            elif cal_model == '朴素贝叶斯分类器':
                from ML_NaiveBayesClassifier import NaiveBayesClassifierx
                st.subheader('朴素贝叶斯分类器|Naive Bayes Classifier')
                st.warning('朴素贝叶斯分类器API:\n'
                '* from sklearn.naive_bayes import GaussianNB\n'
                '* classifier_gaussiannb = GaussianNB()\n'
                '* classifier_gaussiannb.fit(X, y)\n'
                '* y_pred = classifier_gaussiannb.predict(X)\n'
                '* from sklearn import model_selection\n'
                '* X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)\n'
                '* accuracy = model_selection.cross_val_score(classifier_gaussiannb,X, y, scoring="accuracy", cv=num_validations)\n'
                '* f1 = model_selection.cross_val_score(classifier_gaussiannb,X, y, scoring="f1_weighted", cv=num_validations)\n'
                '* precision = model_selection.cross_val_score(classifier_gaussiannb,X, y, scoring="precision_weighted", cv=num_validations)\n'
                '* recall = model_selection.cross_val_score(classifier_gaussiannb,X, y, scoring="recall_weighted", cv=num_validations)\n'
                '---\n'
                '* from ML_NaiveBayesClassifier import NaiveBayesClassifierx\n'
                '* exec(str(NaiveBayesClassifierx()))\n'
                )
                with st.expander('show code'):
                    exec(str(NaiveBayesClassifiercode()))
                st.write('---')
                with st.spinner('please wait...'):
                    exec(str(NaiveBayesClassifierx())) 


            elif cal_model == '高斯朴素贝叶斯分类器':
                from ML_GaussianNBClassifier import GaussianNBClassifierx
                st.subheader('高斯朴素贝叶斯分类器|Gaussian Naive Bayes Classifier')




            
            else:
                nothing = 'do nothing'

        else:
            st.write('wrong page.')



    elif labfunc == '[4]-深度学习模型API':
        st.markdown('coming soon..')



    elif labfunc == '[5]-训练开发工具☆':
        toolfunc = st.sidebar.selectbox('tool-libs:',
        [
            'OCR-光学字符识别',
            'Tree-文件夹结构可视化', 
        ]
        )
        if toolfunc == 'OCR-光学字符识别':
            
            from Tool_OCR import ocrx
            # 2021-10-16 done.
            st.subheader('OCR-Optical Character Recognition，光学字符识别')
            # st.warning('Model-API:') 
            st.warning('ocr(img_path,language)\n* img_path:原始图片路径\n* language:eng:以英语作为识别语言;chi_sim:以中文简体作为识别语言')

            with st.expander('显示配置'):
                st.markdown('项目库:https://github.com/tesseract-ocr/tesseract')
                st.markdown('首先安装<tesseract.exe>:https://github.com/UB-Mannheim/tesseract/wiki')
                st.markdown('安装完成后配置如下环境变量:\n'
                '* Tessearact-OCR添加到Path\n'
                '* 新建系统变量|变量名:TESSDATA_PREFIX，变量值:C:\Program Files\Tesseract-OCR\tessdata')
            
            with st.expander('show code'):
                # from SCodingShow import ocrx1code
                exec(str(ocrx1code()))

            st.markdown('------')
            with st.spinner('Please wait...'):
                exec(str(ocrx()))

        elif toolfunc == 'Tree-文件夹结构可视化':
            # import Tool_FolderTreeStructure
            st.subheader('文件夹路径树结构(Tree Structure)可视化')
            
            with st.expander('show code'):
                exec(str(FolderTreeStructurecode()))
            st.markdown('--------')
            with st.spinner('Please wait...'):
                import Tool_FolderTreeStructure 


        else:
            nothing = 'do nothing' 



    elif labfunc == '[6]-其他':
        other_func = st.sidebar.selectbox('OtherFunc:',
        [
            '[1]-主流库API',
            '[2]-论文期刊',
            '[3]-有用链接',
            '[4]-代码技巧',
        ]
        )

        if other_func == '[1]-主流库API':
            libfunc = st.sidebar.selectbox('hot-libs:',
            [
                'Lib-API-searching',
                'pandas',
                'numpy',
                'matplotlib',
                'seaborn',
                'altair',
                'math',
                'scipy',
                'scikit-learn',
                'tensorflow',
                'keras',
                'cv2',
                'mlflow'
            ]
            )
            
            if libfunc == 'Lib-API-searching':
                sx = st.text_input('Library API Searching cmd')
                st.code('Pandas-pd,Numpy-np,Streamlit-st,')
                st.write('----')
                # st.write(sx)
                cmd = 'st.help('+str(sx)+')'
                if len(sx)>0:
                    exec(cmd)


            elif libfunc == 'pandas':
                st.subheader('Pandas-Cheatsheet')
                st.markdown('https://pandas.pydata.org/docs/reference/index.html')
                # st_display_pdf function not working on web. To be solved.
                # st_display_pdf('docs/pandas.pdf')
                # st_display_pdf('docs/pandas_basics.pdf') 


            elif libfunc == 'numpy':
                st.markdown('x')


            elif libfunc == 'matplotlib':
                st.markdown('x')


            elif libfunc == 'math':
                st.markdown('x')


            elif libfunc == 'scipy':
                st.markdown('x')


            elif libfunc == 'scikit-learn':
                st.markdown('x')


            elif libfunc == 'tensorflow':
                st.markdown('x')


            elif libfunc == 'keras':
                st.markdown('x')


            elif libfunc == 'cv2':
                st.code('pip install opencv-python')


            else:
                nothing = 'do-nothing'


        elif other_func == '[2]-论文期刊':
            st.markdown('coming soon...')


        elif other_func == '[3]-有用链接':
            from UsefulLink import usefullink
            exec(str(usefullink()))
        

        elif other_func == '[4]-代码技巧':
            mfunc = st.sidebar.selectbox('module-funcs:',
            [
                'Python',
                'Git',
                'SQL',
                'Vim'
            ]
            )
        
            if mfunc == 'Python':        
                exec('pythonx()') 
                

            elif mfunc == 'Git':
                exec('gitx()')
        

            elif mfunc == 'Vim':  
                exec('vimx()')


            elif mfunc == 'SQL':
                exec('sqlx()')

            else:
                nothing = 'do nothing'

        else:
            nothing = 'do nothing.'

    else:
        nothing = 'n/a'











# TAI-Lab AccessCode Login Page.
from SessionState import get
session_state = get(AccessCode='')


if session_state.AccessCode != 'tony211':

    head_placeholder = st.empty()
    pwd_placeholder = st.empty()

    head = head_placeholder.subheader('©TAI-Lab')
    pwd = pwd_placeholder.text_input("Access Code", value="", type="password")
    
    session_state.AccessCode = pwd
    if session_state.AccessCode == 'tailab123456':
        head_placeholder.empty()
        pwd_placeholder.empty()
        main()
        # st.sidebar.write('Test passed')

    elif session_state.AccessCode != '':
        st.error("Access denied! Please check https://github.com/tqthooo2021/TAI-Lab or scan **TAI-Lab [QR-Code]** below to get TAI-Lab access code.")
        qrimage = Image.open('pics/QR-code.jpg')
        st.image(qrimage)
    else:
        nothing = 'do nothing'
else:
    main()
    # st.write('Test passed')