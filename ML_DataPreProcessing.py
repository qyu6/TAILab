'''
@func:data pre-processing methods
@create:2021.10.22
'''

def preprocessingx():
    import streamlit as st
    import numpy as np
    from sklearn import preprocessing

    st.write('source-dataset:')
    data = np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
    st.write(data)

    # 标准化处理
    data_standardized = preprocessing.scale(data)
    st.write('\nScaledData',data_standardized)
    st.write("\nScaledMean=",data_standardized.mean(axis=0))
    st.write("\nScaled Std deviation=",data_standardized.std(axis=0))

    # 范围缩放
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data_scaler = data_scaler.fit_transform(data)
    st.write("\nMinMaxScaled data=",data_scaler)

    # 归一化
    data_normalized = preprocessing.normalize(data,norm='l1')
    st.write("\nL1 Normalized data=",data_normalized)

    # 二值化
    data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
    st.write("\nBinarized_data=",data_binarized)

    # 独热编码
    st.write('---')
    encoder = preprocessing.OneHotEncoder()
    data1 = np.array([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
    st.write(data1)
    encoder.fit([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
    encoder_vector = encoder.transform([[2,3,5,3]]).toarray()
    st.write('[2,3,5,3]')
    st.write('\nOneHot Encoded Vector=',encoder_vector)

    # 标记编码: string feature → number
    st.write('---')
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    input_classess = ['audi','ford','audi','toyota','ford','bmw']
    st.write(input_classess)
    label_encoder.fit(input_classess)
    st.write('\nClass-mapping:')
    for i,item in enumerate(label_encoder.classes_):
        st.write(item,'→',i)

    st.write('→Use the label:')
    labels = ['toyota','ford','audi']
    encoded_labels=label_encoder.transform(labels)
    st.write('\nLabels=',labels)
    st.write('\nEncoded labels=',list(encoded_labels))

    st.write('→Back search the label:')
    encoded_labels = [2,1,0,3,1]
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    st.write('\nEncoded labels=',encoded_labels)
    st.write('\nDecoded labels=',decoded_labels)