import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from configs import VERSION
from server.utils import api_address


api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.set_page_config(
        "DSF Chatchat",
        os.path.join("img", "HKUST_favicon.png"),
        initial_sidebar_state="expanded",
        menu_items={
            # 'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            # 'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            'About': f"""欢迎使用 DSF Chat Bot {VERSION}！"""
        }
    )

    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "dsf_lab_logo.jpg"
            ),
            use_column_width=True
        )
        st.text(f"""
                DSF Chat Bot
                支持本地知识库问答
                """)
        st.caption(
            f"""<p align="right">当前版本：{VERSION}</p>""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )
        
        # remove "view full screen" for lab logo image display
        hide_img_fs = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''

        st.markdown(hide_img_fs, unsafe_allow_html=True)

    if selected_page in pages:
        pages[selected_page]["func"](api=api, is_lite=is_lite)
