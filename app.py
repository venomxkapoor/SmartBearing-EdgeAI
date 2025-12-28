import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import generate_data
import time
import json
import os

# Page Config
st.set_page_config(page_title="Industrial Sensor Dashboard", layout="wide")

# Title
st.title("🏭 SmartBearing: Edge-AI Predictive Maintenance")

# --- Sidebar: ROI Calculator ---
st.sidebar.header("ROI Calculator")
downtime_cost = st.sidebar.number_input("Est. Downtime Cost per Hour ($)", min_value=0, value=5000)
hours_saved = st.sidebar.number_input("Est. Hours Saved / Month", min_value=0, value=10)
total_savings = downtime_cost * hours_saved

st.sidebar.metric(label=" projected Monthly Savings", value=f"${total_savings:,.2f}")

st.sidebar.markdown("---")
st.sidebar.info("**System Version**: v1.2 (Edge-Ready)")

# --- Tabs ---
tab_dashboard, tab_validation = st.tabs(["Live Dashboard", "Project Validation (CV Claims)"])

with tab_dashboard:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Sensor Data (FFT Spectrum)")
        
        # Generate a single sample on demand
        if st.button("Generate New Sample"):
            # We'll generate a random type
            s_type = np.random.choice(['normal', 'faulty'])
            raw_signal = generate_data.generate_raw_signal(s_type)
            
            # Process FFT
            fft_signal = generate_data.apply_fft_processing(raw_signal)
            
            # Prepare for Display
            # Create a dataframe for the line chart
            df_chart = pd.DataFrame(fft_signal, columns=["Amplitude"])
            st.line_chart(df_chart)
            
            # --- Prediction & Model Drift ---
            # Load Model (cache this in real app)
            try:
                model = tf.keras.models.load_model('model.keras')
                
                # Preprocess for model (1, 500, 1)
                X_input = fft_signal.reshape(1, len(fft_signal), 1)
                
                # Predict
                prediction = model.predict(X_input, verbose=0)[0][0]
                
                st.divider()
                st.subheader("Diagnostics")
                
                col_pred, col_conf = st.columns(2)
                
                # Interpret
                status = "Faulty" if prediction > 0.5 else "Normal"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                
                with col_pred:
                    st.metric("System Status", status, delta="-Alert" if status == "Faulty" else "Ok")
                    
                with col_conf:
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                # Model Drift / Uncertainty Check
                if 0.4 <= prediction <= 0.6:
                    st.warning("⚠️ Low Confidence Detection (40-60%). Model Retraining Recommended! (Drift Detected)")
                    
            except Exception as e:
                st.error(f"Error loading model: {e}")
                
        else:
            st.info("Click 'Generate New Sample' to analyze sensor data.")

    with col2:
        st.markdown("### Model Architecture")
        st.code("""
Sequential(
  Conv1D(8, ...),
  MaxPool1D,
  Conv1D(16, ...),
  Flatten,
  Dense(16),
  Dense(1, Sigmoid)
)
        """, language="python")

with tab_validation:
    st.header("Project Validation & Benchmarks")
    st.markdown("This section validates the technical claims made in the project validation documentation.")
    
    # helper to load json
    def load_json(filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None

    perf = load_json("performance_metrics.json")
    acc = load_json("accuracy_metrics.json")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.subheader("1. Accuracy Benchmark (CNN vs SVM)")
        if acc:
            st.metric("CNN Accuracy", f"{acc['cnn_accuracy']:.2%}", delta=f"{acc['improvement_pct']:.2f}% vs SVM")
            st.metric("SVM Baseline", f"{acc['svm_accuracy']:.2%}")
        else:
            st.warning("Run 'benchmark_svm.py' to generate metrics.")
            
    with col_v2:
        st.subheader("2. Edge Viability (TFLite)")
        if perf:
            st.metric("Inference Latency", f"{perf['avg_latency_ms']:.2f} ms", delta="< 5ms Target" if perf['latency_target_met'] else "Missed Target")
            st.metric("Model Size", f"{perf['model_size_kb']:.2f} kB", delta="< 50kB Target" if perf['size_target_met'] else "Too Large")
        else:
            st.warning("Run 'verify_performance.py' to generate metrics.")
            
    st.divider()
    st.markdown("**Validation Conclusion:**")
    if perf and perf['latency_target_met'] and perf['size_target_met']:
        st.success("✅ The system meets all Edge Deployment criteria (<5ms latency, <50kB size).")
    else:
        st.warning("⚠️ System optimizations needed to meet all criteria.")

# --- Footer ---
st.markdown("---")
st.caption("SmartBearing IIoT Pipeline | Powered by TensorFlow Lite & Streamlit")
