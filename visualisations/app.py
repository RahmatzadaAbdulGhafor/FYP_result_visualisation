import streamlit as st
import os

from utils import load_metrics, normalize
from radar_chart import plot_radar
from dumbbell_plot import plot_dumbbell
from delta_heatmap import plot_delta_heatmap
from bullet_graph import plot_bullet_graph
from parallel_coordinates import plot_parallel_coordinates

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
BASE_MODELS_DIR = os.path.join(DATA_DIR, "base")
RAG_MODELS_DIR = os.path.join(DATA_DIR, "RAG")

falcon_paths = {
    "4bit": os.path.join(DATA_DIR, "evaluation_results_4bit_falcon_tuned.csv"),
    "8bit": os.path.join(DATA_DIR, "evaluation_results_8bit_falcon_tuned.csv"),
    "fp16": os.path.join(DATA_DIR, "evaluation_results_fp_falcon_tuned.csv")
}





# === Streamlit Setup ===
st.set_page_config(layout="wide", page_title="LLM Visual Dashboard")
st.sidebar.title("📊 Visualisation Selector")
selected_vis = st.sidebar.radio("Choose visualisation", [
    "Dumbbell Plot", "Delta Heatmap", "Parallel Coordinates"
])


st.title("🧠 LLM Evaluation Visual Dashboard")

# === Helper ===
def get_model_names():
    files = os.listdir(BASE_MODELS_DIR)
    names = []
    for f in files:
        if f.endswith(".csv") and "evaluation" in f:
            names.append(f.replace("_evaluation_results.csv", "").replace(".evaluation_results.csv", "").lower())
    return sorted(names)

# === Delta Heatmap (no model selection needed) ===
if selected_vis == "Delta Heatmap":
    st.subheader("📊 Comparing All Models — Delta Heatmap View")
    fig = plot_delta_heatmap(BASE_MODELS_DIR, RAG_MODELS_DIR)
    st.plotly_chart(fig, use_container_width=True)

else:
    # === Model Selector ===
    model_name = st.selectbox("Select a Model", get_model_names())

    base_file = next((f for f in os.listdir(BASE_MODELS_DIR) if model_name in f.lower()), None)
    base_path = os.path.join(BASE_MODELS_DIR, base_file) if base_file else None

    rag_file = next((f for f in os.listdir(RAG_MODELS_DIR) if f.lower().startswith("rag") and model_name in f.lower()), None)
    rag_path = os.path.join(RAG_MODELS_DIR, rag_file) if rag_file else None

    if not base_path or not os.path.exists(base_path):
        st.error(f"Could not find base file for `{model_name}`.")
    elif not rag_path or not os.path.exists(rag_path):
        st.warning(f"RAG version not found for `{model_name}`.")
    else:
        base_metrics = normalize(load_metrics(base_path))
        rag_metrics = normalize(load_metrics(rag_path))

        if selected_vis == "Radar Chart":
            fig = plot_radar(base_metrics, rag_metrics, model_name)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_vis == "Dumbbell Plot":
            if model_name.lower() == "falcon":
                fig1, fig2 = plot_dumbbell(base_metrics, rag_metrics, model_name, falcon_paths=falcon_paths)
            else:
                fig1, fig2 = plot_dumbbell(base_metrics, rag_metrics, model_name)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)


        elif selected_vis == "Bullet Graph":
            fig = plot_bullet_graph(base_metrics, rag_metrics, model_name)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_vis == "Parallel Coordinates":
            csv_path = os.path.join(BASE_DIR, "../data/fpfinetuned - multi_question_resultsfp16.csv")
            fig = plot_parallel_coordinates(csv_path)
            st.plotly_chart(fig, use_container_width=True)