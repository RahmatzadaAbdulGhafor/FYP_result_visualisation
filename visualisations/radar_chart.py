import plotly.graph_objects as go

def plot_radar(base, rag, model_name):
    # Normalize all values to [0–1] for comparability
    all_vals = list(base.values()) + list(rag.values())
    min_val = min(all_vals)
    max_val = max(all_vals)
    def normalize(val): return (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5

    labels = list(base.keys())
    base_vals = [normalize(base[k]) for k in labels]
    rag_vals = [normalize(rag[k]) for k in labels]

    # Close the loop
    base_vals += [base_vals[0]]
    rag_vals += [rag_vals[0]]
    labels += [labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=base_vals,
        theta=labels,
        fill='toself',
        name='Base',
        line=dict(color='royalblue'),
        hoverinfo='text',
        text=[f"{label}: {base[label]:.4f}" for label in labels]
    ))

    fig.add_trace(go.Scatterpolar(
        r=rag_vals,
        theta=labels,
        fill='toself',
        name='RAG',
        line=dict(color='darkorange'),
        hoverinfo='text',
        text=[f"{label}: {rag[label]:.4f}" for label in labels]
    ))

    fig.update_layout(
        title=f"{model_name.capitalize()} — Base vs RAG (Normalized Radar)",
        template='plotly_dark',
        showlegend=True,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                ticks='',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(color='white', size=10),
                gridcolor='rgba(255,255,255,0.1)'
            )
        ),
        legend=dict(
            font=dict(color='white')
        ),
        margin=dict(l=50, r=50, t=60, b=50),
        height=450
    )

    return fig
