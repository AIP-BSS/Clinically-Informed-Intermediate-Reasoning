import os, joblib, sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from typing import Optional

def _slugify(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', s).strip('_')

def plot_roc_with_confidence_interval(
    cvdata: pd.DataFrame,
    true_label_col: str,
    pred_col: str,
    title: str,
    save_path: str,
    n_bootstraps: int = 10000,
    alpha: float = 0.95,
    seed: Optional[int] = 0,
    n_points: int = 200,
    line_color: Optional[str] = None,
    ci_color: Optional[str] = None,
    ci_alpha: float = 0.55,
    diag_color: Optional[str] = None,
):
    y_true = cvdata[true_label_col].to_numpy()
    y_score = cvdata[pred_col].to_numpy()

    base_fpr = np.linspace(0.0, 1.0, n_points)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    rng = np.random.default_rng(seed)
    n = len(cvdata)
    idx = np.arange(n)
    boot_tprs, boot_aucs = [], []
    for _ in range(n_bootstraps):
        sidx = rng.choice(idx, size=n, replace=True)
        yt, ys = y_true[sidx], y_score[sidx]
        if np.unique(yt).size < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yt, ys)
        boot_tprs.append(np.interp(base_fpr, fpr_b, tpr_b))
        boot_aucs.append(roc_auc_score(yt, ys))

    if len(boot_tprs):
        boot_tprs = np.asarray(boot_tprs)
        ci_low_tpr = np.percentile(boot_tprs, (1 - alpha) / 2 * 100, axis=0)
        ci_up_tpr  = np.percentile(boot_tprs, (1 + alpha) / 2 * 100, axis=0)
    else:
        ci_low_tpr = np.interp(base_fpr, fpr, tpr)
        ci_up_tpr  = ci_low_tpr

    if len(boot_aucs):
        auc_lo = float(np.percentile(boot_aucs, (1 - alpha) / 2 * 100))
        auc_hi = float(np.percentile(boot_aucs, (1 + alpha) / 2 * 100))
    else:
        auc_lo, auc_hi = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    plot_kwargs = dict(lw=2)
    if line_color is not None:
        plot_kwargs["color"] = line_color
    (ln,) = ax.plot(fpr, tpr, '-', label=f"ROC (AUC={roc_auc:.3f})", **plot_kwargs)

    base_color = ci_color if ci_color is not None else ln.get_color()
    ax.fill_between(base_fpr, ci_low_tpr, ci_up_tpr, color=base_color, alpha=ci_alpha, edgecolor="none")

    diag_kwargs = dict(lw=1, ls='--')
    if diag_color is not None:
        diag_kwargs["color"] = diag_color
    ax.plot([0, 1], [0, 1], **diag_kwargs)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return float(roc_auc), (auc_lo, auc_hi)

testpath    = "test_sample20.csv"
bundle_path = "bundle_sample.joblib"
intermediate_reasoning_feature     = 'intermediate_reasoning_feature'
save_prediction = False

bundle        = joblib.load(bundle_path)
features_num  = bundle['features_num']
scale_cols    = bundle['scale_cols']
pnlabel       = bundle['pnlabel']
scaler_main   = bundle['scaler_main']
first_ridge_model   = bundle['first_ridge_model']
intermediate_reasoning_feature_scaler = bundle['intermediate_reasoning_feature_scaler']
last_linear_models = bundle['last_linear_models']
test_data = pd.read_csv(testpath)
test_data[scale_cols] = scaler_main.transform(test_data[scale_cols])
add = pd.DataFrame(index=test_data.index)
add[intermediate_reasoning_feature] = first_ridge_model.predict(test_data[features_num])
add[[intermediate_reasoning_feature]] = intermediate_reasoning_feature_scaler.transform(add[[intermediate_reasoning_feature]])
test_data = pd.concat([test_data, add], axis=1)
pred_df = pd.DataFrame(index=test_data.index)
pred_df['case'] = test_data['case'] if 'case' in test_data.columns else pd.Series(range(len(test_data)), index=test_data.index)
pred_df['PN'] = test_data[pnlabel] if pnlabel in test_data.columns else pd.Series(np.nan, index=test_data.index)
models_features_all = {
    'Tabular data of 100 variables directly': features_num,
    'ML-predicted reasoning-oriented score': [intermediate_reasoning_feature],
    'Combination of PSA and ML-predicted reasoning-oriented score': ['PSA', intermediate_reasoning_feature],
}
models_features = {}
for name, feats in models_features_all.items():
    if all(f in test_data.columns for f in feats):
        models_features[name] = feats
    else:
        missing = [f for f in feats if f not in test_data.columns]
        print(f"[WARN] Skip '{name}' (missing columns: {missing})")

for name, feats in models_features.items():
    pred_df[f'{name}_pred'] = last_linear_models[name].predict(test_data[feats])
if pred_df['PN'].notna().all() and pred_df['PN'].nunique() == 2:
    rows = []
    for col in [c for c in pred_df.columns if c.endswith('_pred')]:
        aucval = roc_auc_score(pred_df['PN'], pred_df[col])
        rows.append({'Model': col.replace('_pred',''), 'AUC': aucval})
    pd.DataFrame(rows).to_csv('aucs.csv', index=False)

if save_prediction:
    pred_df.to_csv("predictions.csv", index=False)

if pred_df['PN'].notna().all() and pred_df['PN'].nunique() == 2:
    out_dir = "roc_ci"
    os.makedirs(out_dir, exist_ok=True)

    for col in [c for c in pred_df.columns if c.endswith('_pred')]:
        model_name = col[:-5]
        save_path = os.path.join(out_dir, f"roc_{_slugify(model_name)}.png")
        auc_point, (auc_lo, auc_hi) = plot_roc_with_confidence_interval(
            cvdata=pred_df,
            true_label_col='PN',
            pred_col=col,
            title=model_name,
            save_path=save_path,
            n_bootstraps=100,
            alpha=0.95,
            seed=0,
            n_points=10000,
            line_color="#3c3cff",
            ci_color="#aaaaff",
            ci_alpha=0.55
        )

print("end")



