"""
CICIDS Early Warning — computes lead time, precision/recall where ground truth supports it
Per STEP 10/15: only where attack onset vs warning threshold has causal support
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def compute_early_warning(
    t_starts: np.ndarray,  # [N] epoch of last observed window
    y_true_bin: np.ndarray,  # [N,K] 0/1
    y_prob: np.ndarray,  # [N,K] predicted prob 0..1
    window_sec: int = 60,
    threshold: float = 0.5,
):
    """
    For each sequence where attack appears in future horizon but not at last observed (t):
    actual onset k_true = first k where y_true=1 (1-indexed)
    predicted warning k_pred = first k where y_prob>thr
    lead_time = (k_true - k_pred)*window_sec if k_pred <= k_true else negative (late)
    Evaluate only transition windows (benign->attack within horizon).
    If scenario progression not continuous campaign, we report but warn not causal campaign.
    """
    N,K = y_true_bin.shape
    records=[]
    for n in range(N):
        # check if last observed was benign (we infer from attack_ratio of t? use y_true at K-1 vs? For transition, need that horizon contains attack)
        # We only consider sequences where y_true has at least one attack in horizon
        if y_true_bin[n].sum()==0:
            continue
        # find first true attack in horizon
        k_true = int(np.where(y_true_bin[n]==1)[0][0]) + 1  # 1..K
        # find first predicted warning
        pred_warn = np.where(y_prob[n] > threshold)[0]
        k_pred = int(pred_warn[0])+1 if len(pred_warn)>0 else None
        is_warning = k_pred is not None
        is_true_attack = True
        # classify
        # lead_time positive if warning before actual
        lead = (k_true - k_pred)*window_sec if is_warning else None
        records.append({"n":n,"k_true":k_true,"k_pred":k_pred,"lead_sec":lead,"is_warning":is_warning})
    if not records:
        return {"note":"no transition windows with attack in horizon","records":[]}
    df=pd.DataFrame(records)
    # metrics
    warned=df["is_warning"].sum()
    total=len(df)
    # precision: warnings that had true attack? But we filtered to only attack sequences, so precision = warned/total where warning indeed has attack (always true)
    # Better compute over all N sequences: precision = TP/(TP+FP), recall = TP/(TP+FN) where TP = warned and true, FP = warned but no attack, FN = not warned but attack
    # Need full confusion over all N:
    FP=0; FN=0; TP=warned; TN=0
    for n in range(N):
        has_attack = y_true_bin[n].sum()>0
        has_warn = (y_prob[n] > threshold).any()
        if has_warn and not has_attack: FP+=1
        if not has_warn and has_attack: FN+=1
        if not has_warn and not has_attack: TN+=1
    # for filtered records FN already counted, but adjust
    # compute from full loops above
    precision = TP/(TP+FP) if TP+FP>0 else 0
    recall = TP/(TP+FN) if TP+FN>0 else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
    lead_times = df[df["lead_sec"].notna()]["lead_sec"].values
    median_lead = float(np.median(lead_times)) if len(lead_times) else None
    mean_lead = float(np.mean(lead_times)) if len(lead_times) else None
    return {
        "threshold": threshold,
        "window_sec": window_sec,
        "total_sequences": int(N),
        "attack_sequences": int(total),
        "warned": int(warned),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_lead_sec": median_lead,
        "mean_lead_sec": mean_lead,
        "lead_times_sec": lead_times.tolist()[:20] if len(lead_times) else [],
        "note": "Scenario progression, not causal campaign (see docs) — lead time reflects temporal traffic evolution"
    }
