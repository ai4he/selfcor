import streamlit as st
import math
import numpy as np
import altair as alt
import pandas as pd

# Base-12 Helper Functions (from spec)
def b12_pow12(s):
    return math.pow(12.0, float(s))

def b12_ulp12(scale_digits):
    return math.pow(12.0, -float(scale_digits))

def b12_quantize_nearest_even_digit12(x, scale_digits):
    S = b12_pow12(scale_digits)
    y = x * S
    yi = round(y)  # Ties to even in Python 3
    return yi / S

# Enhanced Base-12 Dot Function with Stats
def b12_dot_ref(x, y, scale=4, early_trip=True):
    A = 0.0
    r12 = 0.0
    c = 0
    ulp = b12_ulp12(scale)
    n = len(x)
    
    residue_history = []
    renorm_events = []
    ops_total = 0
    renorm_local = 0
    
    for i in range(n):
        t = float(x[i]) * float(y[i])
        q = b12_quantize_nearest_even_digit12(t, scale)
        A += q
        r12 += (t - q)
        residue_history.append(abs(r12))
        c = (c + 1) % 12
        ops_total += 1
        if (early_trip and abs(r12) >= 0.5 * ulp) or (c == 0):
            adj = b12_quantize_nearest_even_digit12(r12, scale)
            A += adj
            r12 -= adj
            renorm_events.append(i)
            renorm_local += 1
    
    # Final fold
    if abs(r12) > 0.0:
        adj = b12_quantize_nearest_even_digit12(r12, scale)
        A += adj
        renorm_local += 1
    
    stats = {
        'ops_total': ops_total,
        'renorm_local': renorm_local,
        'max_abs_r12': max(residue_history) if residue_history else 0,
    }
    
    return A, stats, residue_history, renorm_events

# Streamlit App
st.title("Base-12 12-RRC Demo: Self-Correcting Dot Product")

st.markdown("""
This web demo implements a preliminary Base-12 12-RRC dot product from the spec. 
Adjust parameters below and run to see results and the Drift Dashboard.
""")

# User Inputs
vector_size = st.slider("Vector Size (Multiple of 12 for full cycles)", min_value=12, max_value=120, value=24, step=12)
scale_digits = st.slider("Scale Digits (Fractional base-12 digits)", min_value=3, max_value=6, value=4)
early_trip = st.checkbox("Enable Early Trip (Threshold-based renorm)", value=True)

if st.button("Run Demo"):
    # Generate sample data
    x = np.linspace(0.1, vector_size / 10.0, vector_size)
    y = np.ones(vector_size)
    
    # Compute
    std_dot = np.dot(x, y)
    b12_result, stats, residue_history, renorm_events = b12_dot_ref(x, y, scale=scale_digits, early_trip=early_trip)
    
    # Display Results
    st.subheader("Results")
    st.write(f"Standard NumPy Dot: {std_dot}")
    st.write(f"Base-12 RRC Dot: {b12_result}")
    st.write(f"Error: {abs(std_dot - b12_result)}")
    st.write(f"Stats: {stats}")
    
    # Prepare Data for Altair Charts
    residue_df = pd.DataFrame({'Operation Index': range(len(residue_history)), '|r12|': residue_history})
    renorm_df = pd.DataFrame({'Renorm Events': renorm_events, 'y': [max(residue_history or [0])] * len(renorm_events)})
    hist_df = pd.DataFrame({'|r12|': residue_history})
    bar_df = pd.DataFrame({'Category': ['Local Renorms'], 'Count': [stats['renorm_local']]})
    error_df = pd.DataFrame({'Category': ['Error'], 'Value': [abs(std_dot - b12_result)]})
    
    # Dashboard Charts (using Altair)
    st.subheader("Drift Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        # Residue Timeline
        timeline = alt.Chart(residue_df).mark_line(color='blue').encode(
            x='Operation Index',
            y='|r12|',
            tooltip=['Operation Index', '|r12|']
        ).properties(title='Residue Magnitude Timeline')
        events = alt.Chart(renorm_df).mark_rule(color='red', strokeDash=[4,4]).encode(
            x='Renorm Events'
        )
        st.altair_chart(timeline + events, use_container_width=True)
    
    with col2:
        # Residue Histogram
        hist = alt.Chart(hist_df).mark_bar(color='skyblue').encode(
            alt.X('|r12|', bin=True),
            y='count()',
            tooltip=['|r12|', 'count()']
        ).properties(title='Residue Magnitude Histogram')
        st.altair_chart(hist, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        # Renorm Frequency
        renorm_bar = alt.Chart(bar_df).mark_bar(color='green').encode(
            x='Category',
            y='Count',
            tooltip=['Category', 'Count']
        ).properties(title='Renorm Frequency')
        st.altair_chart(renorm_bar, use_container_width=True)
    
    with col4:
        # Error vs Truth
        error_bar = alt.Chart(error_df).mark_bar(color='orange').encode(
            x='Category',
            y='Value',
            tooltip=['Category', 'Value']
        ).properties(title='Error vs FP Truth')
        st.altair_chart(error_bar, use_container_width=True)
