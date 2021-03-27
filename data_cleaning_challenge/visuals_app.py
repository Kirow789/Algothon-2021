import pandas as pd 
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.write("""
# Algothon Data Visualisation App
""")

df = pd.read_csv('GruzenMachen_data_cleaning.csv')

fast = 20
slow = 200

slow_MA = df.ewm(span=slow).mean()
fast_MA = df.ewm(span=fast).mean()

x = [i for i in range(len(df))]
y = np.array([df.iloc[i][0] for i in range(len(df))])
y_fast = np.array([fast_MA.iloc[i][0] for i in range(len(fast_MA))])
y_slow = np.array([slow_MA.iloc[i][0] for i in range(len(slow_MA))])

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='Series 1'))
fig.add_trace(go.Scatter(x=x, y=y_fast,
                    mode='lines',
                    name='SMA20'))
fig.add_trace(go.Scatter(x=x, y=y_slow,
                    mode='lines',
                    name='SMA200'))

st.write(fig)




# fig = px.line(df, width=1000, height=600)
# fig.show()

# plots = []
# plots.append(go.Scatter(df))

# fig = go.Figure(plots)

# st.plotly_chart(fig)




# st.line_chart(df)
# st.line_chart(fast_MA)
# st.line_chart(slow_MA)

# fig = px.line(df)
# fig.add_scatter(px.line(fast_MA))
# st.write(fig)

# Create traces
