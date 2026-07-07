import streamlit as st
from pathlib import Path
import sys

import numpy as np
import torch
import streamlit as st

st.write("NumPy:", np.__version__)
st.write("Torch:", torch.__version__)

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