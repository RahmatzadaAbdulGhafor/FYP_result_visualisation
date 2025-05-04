import pandas as pd
import plotly.graph_objects as go
import ast

def extract_fre(read_str):
    try:
        d = ast.literal_eval(read_str)
        return d.get("flesch_reading_ease")
    except:
        return None

def plot_parallel_coordinates(csv_path):
    df = pd.read_csv(csv_path)

    # Extract and clean readability
    df["Readability (FRE)"] = df["Readability"].apply(extract_fre)

    # Selected and renamed metrics
    selected_cols = {
        "BLEU": "BLEU",
        "ROUGE-1": "ROUGE-1 F1",
        "ROUGE-2": "ROUGE-2 F1",
        "ROUGE-L": "ROUGE-L F1",
        "BERTScore": "BERTScore",
        "F1": "F1",
        "Readability (FRE)": "Readability (FRE)"
    }

    df_selected = df[list(selected_cols.keys())].rename(columns=selected_cols).dropna()

    # Min-max normalize
    for col in df_selected.columns:
        min_val = df_selected[col].min()
        max_val = df_selected[col].max()
        df_selected[col] = (df_selected[col] - min_val) / (max_val - min_val) if min_val != max_val else 0.5

    # Plot
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df_selected["F1"],
            colorscale="Tealrose",
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="F1 Score", font=dict(size=14))
            )
        ),
        dimensions=[
            dict(label="BLEU", values=df_selected["BLEU"]),
            *[
                dict(label=col, values=df_selected[col], tickvals=[], ticktext=[])
                for col in df_selected.columns
                if col not in ["BLEU", "F1", "Prompt"]
            ]
        ]
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Parallel Coordinates Plot — Finetuned Model Metrics",
        title_font_size=22,
        margin=dict(t=90, l=80, r=80, b=60),
        height=600
    )

    return fig
