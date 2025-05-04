import pandas as pd
import re
import ast

def extract_rouge1_f1(rouge_str):
    import re
    if not isinstance(rouge_str, str):
        return None
    match = re.search(r"rouge1.*?fmeasure=([\d.]+)", rouge_str)
    return float(match.group(1)) if match else None

def extract_fre(read_str):
    import ast
    try:
        d = ast.literal_eval(read_str)
        return d.get("flesch_reading_ease")
    except:
        return None


def safe_mean(series):
    return pd.to_numeric(series, errors='coerce').mean()

def load_metrics(path):
    df = pd.read_csv(path)

    # Drop rows with missing ROUGE or Perplexity values
    df = df.dropna(subset=["ROUGE", "Perplexity"])

    return {
        "Perplexity": safe_mean(df["Perplexity"]),
        "BLEU": safe_mean(df["BLEU"]),
        "ROUGE-1 F1": df["ROUGE"].apply(extract_rouge1_f1).mean(),
        "BERTScore": safe_mean(df["BERTScore"]),
        "F1": safe_mean(df["F1"]),
        "Readability (FRE)": df["Readability"].apply(extract_fre).mean(),
        "Latency": safe_mean(df["Latency"]),
        "Throughput": safe_mean(df["Throughput (tokens/sec)"]),
        "Hallucination": safe_mean(df.get("Hallucination", pd.Series([None])))
    }

def normalize(metrics):
    norm = {}
    for k, v in metrics.items():
        if v is None:
            norm[k] = 0
        elif k in ["Perplexity", "Latency", "Hallucination"]:
            norm[k] = -v  # Lower is better
        else:
            norm[k] = v   # Higher is better
    return norm

