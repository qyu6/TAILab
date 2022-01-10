'''
@func: open multiple website onetime.
'''

import webbrowser
websites = [
    'https://share.streamlit.io/tqthooo2021/tai-lab/main/labmain.py',
    'https://docs.streamlit.io/library/api-reference',
    'https://mp.weixin.qq.com/',
    'https://www.zhihu.com/pub/reader/119564904/chapter/975098904496365568',
    'https://www.deepl.com/translator#zh/en/%E5%BC%80%E5%8F%91%E6%97%A5%E5%BF%97',
    'https://github.com/tqthooo2021/TAI-Lab',
]
for url in websites:
    webbrowser.open(url)