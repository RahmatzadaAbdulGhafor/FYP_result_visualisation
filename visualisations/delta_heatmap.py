import pandas as pd
import plotly.express as px
import os
from utils import load_metrics

def plot_delta_heatmap(base_dir, rag_dir):
    import numpy as np
    delta_data = {}

    for filename in os.listdir(base_dir):
        if not filename.endswith(".csv"):
            continue

        model = filename.replace("_evaluation_results.csv", "").replace(".evaluation_results.csv", "").lower()
        base_path = os.path.join(base_dir, filename)

        # Try to find matching RAG file
        rag_match = next((f for f in os.listdir(rag_dir)
                          if f.lower().startswith("rag") and model in f.lower()), None)
        if not rag_match:
            continue

        rag_path = os.path.join(rag_dir, rag_match)
        base_metrics = load_metrics(base_path)
        rag_metrics = load_metrics(rag_path)

        delta = {
            k: rag_metrics[k] - base_metrics[k]
            if base_metrics[k] is not None and rag_metrics[k] is not None
            else None
            for k in base_metrics
        }

        delta_data[model] = delta

    # ✅ ADD FINETUNED FALCON (first row only)
    finetuned_path = os.path.join(base_dir, "../fpfinetuned - multi_question_resultsfp16.csv")
    finetuned_df = pd.read_csv(finetuned_path)
    first_row = finetuned_df.iloc[0]

    # Extract from that row only
    falcon_base_metrics = load_metrics(os.path.join(base_dir, "falcon_evaluation_results.csv"))

    finetuned_metrics = {
        "Perplexity": None,  # Not available
        "BLEU": first_row["BLEU"],
        "ROUGE-1 F1": first_row["ROUGE-1"],
        "BERTScore": first_row["BERTScore"],
        "F1": first_row["F1"],
        "Readability (FRE)": eval(first_row["Readability"]).get("flesch_reading_ease", None),
        "Latency": None,  # Not available
        "Throughput": None,  # Not available
        "Hallucination": None  # Not available
    }

    # Compute deltas
    delta_ft = {
        k: finetuned_metrics[k] - falcon_base_metrics[k]
        if finetuned_metrics[k] is not None and falcon_base_metrics[k] is not None
        else None
        for k in falcon_base_metrics
    }

    delta_data["falcon_finetuned"] = delta_ft

    # Make heatmap
    df = pd.DataFrame(delta_data).T  # models as rows

    fig = px.imshow(
        df,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        zmin=-df.abs().max().max(),
        zmax=df.abs().max().max(),
        labels=dict(x="Metric", y="Model", color="Δ (RAG - Base)"),
        aspect="auto"
    )

    fig.update_layout(
        title="Delta Change Heatmap — RAG vs Base Models (including Falcon Finetuned)",
        template="plotly_dark",
        xaxis=dict(side="top", tickangle=45),
        margin=dict(t=100, l=100, r=20, b=20),
        height=600
    )

    return fig
