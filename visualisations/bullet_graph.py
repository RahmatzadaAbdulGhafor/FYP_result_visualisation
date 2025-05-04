import plotly.graph_objects as go
import pandas as pd

def plot_bullet_graph(base_metrics, rag_metrics, model_name):
    df = pd.DataFrame({
        "Metric": list(base_metrics.keys()),
        "Base": list(base_metrics.values()),
        "RAG": list(rag_metrics.values())
    })

    df = df.sort_values("Base", ascending=False)

    fig = go.Figure()

    # Add only one legend entry for each type
    legend_added = {"Reference Scale": False, "Base": False, "RAG": False}

    for i, row in df.iterrows():
        metric = row["Metric"]
        base_val = row["Base"]
        rag_val = row["RAG"]

        if base_val is None or rag_val is None:
            continue

        max_val = max(abs(base_val), abs(rag_val)) * 1.4
        buffer = max_val * 0.1

        # Background bar (reference scale)
        fig.add_trace(go.Bar(
            x=[max_val],
            y=[metric],
            marker=dict(color="lightgray"),
            orientation='h',
            width=0.4,
            name="Reference Scale" if not legend_added["Reference Scale"] else "",
            showlegend=not legend_added["Reference Scale"],
            hoverinfo='skip'
        ))
        legend_added["Reference Scale"] = True

        # Base bar
        fig.add_trace(go.Bar(
            x=[base_val],
            y=[metric],
            marker=dict(color="royalblue"),
            orientation='h',
            width=0.2,
            name="Base" if not legend_added["Base"] else "",
            showlegend=not legend_added["Base"],
            hovertemplate="Base: %{x:.4f}<extra></extra>"
        ))
        legend_added["Base"] = True

        # RAG marker
        # RAG marker (more visible fix)
        fig.add_trace(go.Scatter(
            x=[rag_val],
            y=[metric],
            mode='markers',
            marker=dict(color="orange", size=14, symbol="diamond"),
            name="RAG" if not legend_added["RAG"] else "",
            hovertemplate="RAG: %{x:.4f}<extra></extra>",
            showlegend=not legend_added["RAG"]
        ))

        legend_added["RAG"] = True

    fig.update_layout(
        title=f"{model_name.capitalize()} — Bullet Graph Comparison (Base vs RAG)",
        barmode='overlay',
        template='plotly_dark',
        xaxis=dict(
            title='Metric Value',
            showgrid=True,
            zeroline=False,
            rangemode='tozero'
        ),
        yaxis=dict(
            title='',
            tickmode='array',
            tickvals=df["Metric"],
            autorange='reversed'
        ),
        margin=dict(t=60, l=100, r=40, b=40),
        height=500,
        legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
