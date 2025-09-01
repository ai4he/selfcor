import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Create Dashboard Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Base-12 12-RRC Drift Dashboard', fontsize=16)
    
    # Residue Timeline
    axs[0, 0].plot(residue_history, label='|r12|')
    axs[0, 0].vlines(renorm_events, ymin=0, ymax=max(residue_history or [0]), color='r', linestyle='--', label='Renorm Events')
    axs[0, 0].set_title('Residue Magnitude Timeline')
    axs[0, 0].set_xlabel('Operation Index')
    axs[0, 0].set_ylabel('|r12|')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Residue Histogram
    axs[0, 1].hist(residue_history, bins=10, color='skyblue', edgecolor='black')
    axs[0, 1].set_title('Residue Magnitude Histogram')
    axs[0, 1].set_xlabel('|r12| Bins')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].grid(True)
    
    # Renorm Frequency
    axs[1, 0].bar(['Local Renorms'], [stats['renorm_local']], color='green')
    axs[1, 0].set_title('Renorm Frequency')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].grid(True)
    
    # Error vs Truth
    axs[1, 1].bar(['Error'], [abs(std_dot - b12_result)], color='orange')
    axs[1, 1].set_title('Error vs FP Truth')
    axs[1, 1].set_ylabel('Absolute Error')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
