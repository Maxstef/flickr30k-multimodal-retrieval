import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Mini-CLIP Explorer",
    page_icon="🔎",
    layout="wide",
)

st.title("Mini-CLIP Explorer")

st.write("Select a page from the sidebar to start exploring the model.")

st.switch_page("pages/01_home.py")
