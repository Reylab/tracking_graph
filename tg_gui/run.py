import os
def run_tg_server():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'tg_gui.py')
    os.system("streamlit run "+filename)
