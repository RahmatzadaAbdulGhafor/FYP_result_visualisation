import plotly.graph_objects as go
import pandas as pd
import os
import ast
from collections import namedtuple
Score = namedtuple("Score", ["precision", "recall", "fmeasure"])


def load_metrics_from_csv(path):
    
    df = pd.read_csv(path)

    # Extract ROUGE if needed
    if "ROUGE-1 F1" not in df.columns and "ROUGE" in df.columns:
        def extract_rouge_f1(rstr):
            try:
                r = eval(rstr)  # Use eval to preserve Score object
                return {
                    "ROUGE-1 F1": r["rouge1"].fmeasure,
                    "ROUGE-2 F1": r["rouge2"].fmeasure,
                    "ROUGE-L F1": r["rougeL"].fmeasure
                }
            except Exception as e:
                print(f"⚠️ ROUGE parse error: {e}")
                return {"ROUGE-1 F1": None, "ROUGE-2 F1": None, "ROUGE-L F1": None}


        rouge_f1s = df["ROUGE"].apply(extract_rouge_f1).apply(pd.Series)
        df = pd.concat([df, rouge_f1s], axis=1)

    # Extract Readability FRE
    def extract_fre(val):
        try:
            return ast.literal_eval(val).get("flesch_reading_ease", None)
        except:
            return None

    if "Readability (FRE)" not in df.columns:
        df["Readability (FRE)"] = df["Readability"].apply(extract_fre)

    # Coerce numerics safely
    for col in ["Latency", "Perplexity", "Throughput (tokens/sec)", "F1", "BLEU", "BERTScore",
                "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "Readability (FRE)", "Hallucination"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select final metrics
    selected = {}
    for col in [
        "BLEU", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore", "F1",
        "Readability (FRE)", "Latency", "Throughput (tokens/sec)", "Perplexity", "Hallucination"
    ]:
       selected[col] = df[col].dropna().mean() if col in df.columns else None

    # print("✔️ Latency column dtype:", df["Latency"].dtype)
    # print("✔️ Latency mean:", df["Latency"].mean())
    print("📊 Final Metrics for", path)
    for k, v in selected.items():
        print(f"{k}: {v} ({type(v)})")

    return selected


def plot_dumbbell(base_metrics, rag_metrics, model_name, falcon_paths=None):
    df = pd.DataFrame({
        "Metric": list(base_metrics.keys()),
        "Base": list(base_metrics.values()),
        "RAG": list(rag_metrics.values())
    })

    df["Delta"] = df["RAG"] - df["Base"]
    df = df.sort_values("Delta")

    small_range_metrics = ["BLEU", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "F1", "BERTScore"]
    large_range_metrics = ["Readability (FRE)", "Latency", "Throughput (tokens/sec)", "Perplexity", "Hallucination"]

    # Default Falcon model file paths if not passsed 
    # if falcon_paths is None:
    #     falcon_paths = {
    #     "4bit": os.path.join("..", "data", "evaluation_results_4bit_falcon_tuned.csv"),
    #     "8bit": os.path.join("..", "data", "evaluation_results_8bit_falcon_tuned.csv"),
    #     "fp16": os.path.join("..", "data", "evaluation_results_fp_falcon_tuned.csv"),

    # }


    falcon_colors = {
        "4bit": "green",
        "8bit": "red",
        "fp16": "purple"
    }

    def make_fig(filtered_df, title_suffix, skip_metric_for_falcon=None):
        

        fig = go.Figure()

        # Create numeric y-axis map
        metric_names = list(filtered_df["Metric"])
        y_map = {metric: i for i, metric in enumerate(metric_names)}

        # Dumbbell lines: Base to RAG
        for _, row in filtered_df.iterrows():
            y_val = y_map[row["Metric"]]
            fig.add_trace(go.Scatter(
                x=[row["Base"], row["RAG"]],
                y=[y_val, y_val],
                mode='lines',
                line=dict(color='lightgray', width=3),
                showlegend=False
            ))

        # Plot Base and RAG points
        fig.add_trace(go.Scatter(
            x=filtered_df["Base"],
            y=[y_map[m] for m in filtered_df["Metric"]],
            mode='markers',
            marker=dict(size=10, color='royalblue'),
            name='Base',
            hovertemplate='Base: %{x:.4f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=filtered_df["RAG"],
            y=[y_map[m] for m in filtered_df["Metric"]],
            mode='markers',
            marker=dict(size=10, color='darkorange'),
            name='RAG',
            hovertemplate='RAG: %{x:.4f}<extra></extra>'
        ))

        if falcon_paths is not None:
        # Falcon-specific markers with stsggerdde y-axis
            falcon_offsets = {"4bit": -0.2, "8bit": 0, "fp16": 0.2}
            for label, path in falcon_paths.items():
                if os.path.exists(path):
                    metrics = load_metrics_from_csv(path)
                    for metric in filtered_df["Metric"]:
                        if skip_metric_for_falcon and metric == skip_metric_for_falcon:
                            continue
                        if metric in metrics and metrics[metric] is not None:
                            y_val = y_map[metric] + falcon_offsets[label]
                            fig.add_trace(go.Scatter(
                                x=[metrics[metric]],
                                y=[y_val],
                                showlegend=(metric == filtered_df["Metric"].iloc[0]),
                                name=f"Fine-Tuned Falcon ({label})",
                                marker=dict(
                                    size=10,
                                    color=falcon_colors[label],
                                    line=dict(width=1, color='black')
                                ),
                                hovertemplate=f"Falcon ({label}): %{{x:.4f}}<extra></extra>"
                            ))
                else:
                    print(f"file not found: {path}")

        # Axis range buffer
        all_vals = pd.concat([filtered_df["Base"], filtered_df["RAG"]])
        x_min = all_vals.min()
        x_max = all_vals.max()
        buffer = (x_max - x_min) * 0.1 if x_max != x_min else 0.1

        # Final layout
        fig.update_layout(
            title=f"{model_name.capitalize()} — {title_suffix}",
            xaxis_title="Metric Value",
            yaxis=dict(
                tickmode='array',
                tickvals=list(y_map.values()),
                ticktext=list(y_map.keys()),
            ),
            template="simple_white",
            margin=dict(l=50, r=150, t=60, b=40),
            height=350 + len(filtered_df) * 20,
            xaxis=dict(range=[x_min - buffer, x_max + buffer])
        )

        return fig  


    fig1 = make_fig(df[df["Metric"].isin(small_range_metrics)], "Quality Metrics (Zoomed In)")

    if model_name.lower() == "falcon":
        filtered_df = df[df["Metric"].isin(large_range_metrics) & (df["Metric"] != "Hallucination")]
        fig2 = make_fig(filtered_df, "Performance & Others", skip_metric_for_falcon="Hallucination")
    else:
        fig2 = make_fig(df[df["Metric"].isin(large_range_metrics)], "Performance & Others")

    return fig1, fig2
