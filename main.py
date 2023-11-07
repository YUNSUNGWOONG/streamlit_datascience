import streamlit as st
import pandas as pd

view = [100, 150, 30]
sview = pd.Series(view)
st.write('### ※This is pure data')
st.write(view)
st.write("""
### ※This is data processed by pandas
### &nbsp;&nbsp;&nbsp;[This is data that pandas processes]
""")
st.write(sview)