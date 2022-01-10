'''
@func:OCR model
@create:2021.10.27
'''
def ocrx():
    import streamlit as st
    from PIL import Image

    # OCR显示代码：
    image = Image.open('pics/ocr_test.png')
    st.image(image)

    st.markdown('Model-Output:')
    col1,col2 = st.columns(2)
    col1.code('''
ocr('pics/ocr_test.png','eng')
********************************
B® WD -H-KANFARAA PEAS
Be: Gex@ora) | 94:(FV7162X ATG!
ah LFV2A2156C03333333
S:
A 1845 ko RRAR:5) A
AHH: (2012/06/06
RAMAS: | CLS | WewF:| 77 ew HH:1598 mi
    ''')

    col2.code('''
ocr('pics/ocr_test.png','chi_sim')
********************************
G思 @ 一上一大众汽车有限公司"中国负寺
商标: 。 葵末(BORA) | 型号: [F V7162XATG
0   LFV2A2156C3333333
入:
总质量:[18 4 5jkg科几人数:|5|人
制造日期: |2012/06/06
发动机型号: | CL SS | 额定功率:| 77 jkW 排量:1598jml
    ''')