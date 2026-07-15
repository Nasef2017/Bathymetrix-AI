import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from qgis.core import (
    QgsCoordinateTransform,
    QgsProcessingException,
    QgsProject,
    QgsRasterLayer,
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def extract_values(vec_layer, depth_field, p3_path, p4_path, feedback):
    has_p4 = p4_path is not None and str(p4_path).strip() != "" and p4_path != "None" and os.path.exists(p4_path) and p4_path != p3_path

    src4 = None
    try:
        src3 = rasterio.open(p3_path)
        band3 = src3.read(1).astype(np.float32)
        nodata3 = src3.nodata
        height = src3.height
        width = src3.width

        if has_p4:
            src4 = rasterio.open(p4_path)
            band4 = src4.read(1).astype(np.float32)
            nodata4 = src4.nodata
    except Exception as e:
        raise QgsProcessingException(f"Failed to open raster: {e}")

    raster_crs = QgsRasterLayer(p3_path).crs()
    tr = QgsCoordinateTransform(
        vec_layer.sourceCrs(), raster_crs, QgsProject.instance()
    )

    obs, pred3, pred4 = [], [], []

    for feat in vec_layer.getFeatures():
        raw_depth = feat[depth_field]
        if raw_depth is None:
            continue

        if not feat.hasGeometry():
            continue

        geom = feat.geometry()
        if geom.isNull():
            continue

        try:
            geom.transform(tr)
            pt = geom.asPoint()
        except Exception:  # nosec B112
            continue

        try:
            row, col = src3.index(pt.x(), pt.y())
        except Exception:  # nosec B112
            continue

        if not (0 <= row < height and 0 <= col < width):
            continue

        v3 = float(band3[row, col])
        if nodata3 is not None and v3 == nodata3:
            continue
        if not np.isfinite(v3):
            continue
        if v3 <= -9990:
            continue

        v4 = None
        if has_p4 and src4 is not None:
            v4 = float(band4[row, col])
            if nodata4 is not None and v4 == nodata4:
                continue
            if not np.isfinite(v4):
                continue
            if v4 <= -9990:
                continue

        obs.append(float(raw_depth))
        pred3.append(v3)
        if has_p4:
            pred4.append(v4)

    src3.close()
    if src4 is not None:
        src4.close()

    return np.array(obs), np.array(pred3), (np.array(pred4) if has_p4 else None)


def calc_stats(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))

    sum_abs_diff = np.sum(np.abs(y_true - y_pred))
    sum_abs_true = np.sum(np.abs(y_true))
    wmape = (sum_abs_diff / sum_abs_true * 100) if sum_abs_true != 0 else 0.0

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias, "wMAPE": wmape}


def compute_iho_details(rmse, mean_depth):
    d = float(abs(mean_depth))
    # S-44 TVU Max Formulas
    tvu_special = np.sqrt(0.25**2 + (0.0075 * d)**2)
    tvu_order1 = np.sqrt(0.5**2 + (0.013 * d)**2)
    tvu_order2 = np.sqrt(1.0**2 + (0.023 * d)**2)
    
    model_tvu = 1.96 * rmse
    
    if model_tvu <= tvu_special:
        achieved = "Special Order"
        limit = tvu_special
    elif model_tvu <= tvu_order1:
        achieved = "Order 1a/1b"
        limit = tvu_order1
    elif model_tvu <= tvu_order2:
        achieved = "Order 2"
        limit = tvu_order2
    else:
        achieved = "Out of Spec"
        limit = tvu_order2
        
    return achieved, model_tvu, limit


def stratified_analysis(y_true, y_pred, model_name):
    bins = [0, 5, 10, 15, 20, 30, 50, 100]
    labels = [f"{bins[i]}-{bins[i + 1]}m" for i in range(len(bins) - 1)]

    abs_true = np.abs(y_true)
    rows = []

    s = calc_stats(y_true, y_pred)
    mean_d = float(np.mean(abs_true)) if len(abs_true) > 0 else 0.0
    achieved, model_tvu, limit = compute_iho_details(s["RMSE"], mean_d)
    
    if achieved == "Special Order":
        uses = "Navigational safety, under-keel clearance, dredging, harbor engineering, benthic habitat modeling."
    elif achieved == "Order 1a/1b":
        uses = "General coastal shipping, marine spatial planning, coastal erosion assessment, habitat mapping, wave modeling."
    elif achieved == "Order 2":
        uses = "General bathymetric mapping, pre-survey planning, tsunami/tide modeling, marine resource exploration."
    else:
        uses = "Reconnaissance surveys, preliminary coastal planning, general environmental visualization, low-risk modeling."

    rows.append({
        "Model": model_name,
        "Depth_Bin": "GLOBAL",
        "Count": len(y_true),
        "Mean_Depth": round(mean_d, 2),
        **s,
        "Model_TVU_95": round(model_tvu, 3),
        "IHO_TVU_Limit": round(limit, 3),
        "IHO_Order": achieved,
        "Suggested_Uses": uses
    })

    for i in range(len(bins) - 1):
        mask = (abs_true >= bins[i]) & (abs_true < bins[i + 1])
        n = int(np.sum(mask))
        if n >= 5:
            sb = calc_stats(y_true[mask], y_pred[mask])
            mean_d_bin = float(np.mean(abs_true[mask]))
            achieved_bin, model_tvu_bin, limit_bin = compute_iho_details(sb["RMSE"], mean_d_bin)
            
            if achieved_bin == "Special Order":
                uses_bin = "Navigational safety, under-keel clearance, dredging, harbor engineering, benthic habitat modeling."
            elif achieved_bin == "Order 1a/1b":
                uses_bin = "General coastal shipping, marine spatial planning, coastal erosion assessment, habitat mapping, wave modeling."
            elif achieved_bin == "Order 2":
                uses_bin = "General bathymetric mapping, pre-survey planning, tsunami/tide modeling, marine resource exploration."
            else:
                uses_bin = "Reconnaissance surveys, preliminary coastal planning, general environmental visualization, low-risk modeling."

            rows.append({
                "Model": model_name,
                "Depth_Bin": labels[i],
                "Count": n,
                "Mean_Depth": round(mean_d_bin, 2),
                **sb,
                "Model_TVU_95": round(model_tvu_bin, 3),
                "IHO_TVU_Limit": round(limit_bin, 3),
                "IHO_Order": achieved_bin,
                "Suggested_Uses": uses_bin
            })

    return rows


def write_final_verdict(out_dir, s3, s4, n_val, p3_name, p4_name):
    has_p4 = s4 is not None

    if not has_p4:
        winner = "Phase 03 (Global Model)"
        reason = "Phase 04 was disabled by the user."
    else:
        imp_rmse = ((s3["RMSE"] - s4["RMSE"]) / s3["RMSE"]) * 100 if s3["RMSE"] != 0 else 0
        imp_wmape = (
            ((s3["wMAPE"] - s4["wMAPE"]) / s3["wMAPE"]) * 100 if s3["wMAPE"] != 0 else 0
        )
        imp_r2 = (s4["R2"] - s3["R2"]) * 100

        if np.isclose(s4["RMSE"], s3["RMSE"], atol=0.01):
            winner = "TIE — No significant difference between models"
            reason = "RMSE difference is within 0.01 m tolerance."
        elif s4["RMSE"] < s3["RMSE"]:
            winner = "Phase 04 (Refined / Best Map)"
            reason = "Lower RMSE confirms improvement from adaptive refinement."
        else:
            winner = "Phase 03 (Global Model)"
            reason = "Phase 04 did not improve RMSE — possible overfitting in adaptive step."

    sep = "=" * 65
    lines = [
        sep,
        "           SDB FINAL SCIENTIFIC VALIDATION REPORT          ",
        sep,
        f"  Phase 03 map : {p3_name}",
    ]
    if has_p4:
        lines.append(f"  Phase 04 map : {p4_name}")
    lines.extend([
        f"  Validation points used : {n_val}",
        "",
        "--- METRICS COMPARISON (VALIDATION SET) ---",
    ])

    if has_p4:
        lines.extend([
            f"{'Metric':<12} {'Phase 03 (Global)':>20} {'Phase 04 (Refined)':>20} {'Improvement':>15}",
            "-" * 70,
            f"{'RMSE (m)':<12} {s3['RMSE']:>20.4f} {s4['RMSE']:>20.4f} {imp_rmse:>+14.2f}%",
            f"{'R²':<12} {s3['R2']:>20.4f} {s4['R2']:>20.4f} {imp_r2:>+14.2f} pts",
            f"{'MAE (m)':<12} {s3['MAE']:>20.4f} {s4['MAE']:>20.4f}",
            f"{'Bias (m)':<12} {s3['Bias']:>20.4f} {s4['Bias']:>20.4f}",
            f"{'wMAPE (%)':<12} {s3['wMAPE']:>20.2f} {s4['wMAPE']:>20.2f} {imp_wmape:>+14.2f}%",
        ])
    else:
        lines.extend([
            f"{'Metric':<12} {'Phase 03 (Global)':>20}",
            "-" * 40,
            f"{'RMSE (m)':<12} {s3['RMSE']:>20.4f}",
            f"{'R²':<12} {s3['R2']:>20.4f}",
            f"{'MAE (m)':<12} {s3['MAE']:>20.4f}",
            f"{'Bias (m)':<12} {s3['Bias']:>20.4f}",
            f"{'wMAPE (%)':<12} {s3['wMAPE']:>20.2f}",
        ])

    lines.extend([
        "",
        "--- FINAL VERDICT ---",
        f"  WINNER : {winner}",
        f"  REASON : {reason}",
        "",
        "NOTE: wMAPE = Sum(|Error|) / Sum(|Observed|) × 100",
        "      Negative depths are handled correctly (absolute values used in bins).",
        sep,
    ])

    report_path = os.path.join(out_dir, "5_FINAL_SUMMARY.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def _axis_limits(*arrays):
    all_vals = np.concatenate([a.ravel() for a in arrays])
    lo = np.nanmin(all_vals)
    hi = np.nanmax(all_vals)
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _subplot_scatter(ax, obs, pred, stats, title, use_kde):
    if use_kde:
        try:
            xy = np.vstack([obs, pred])
            z = gaussian_kde(xy)(xy)
            sc = ax.scatter(obs, pred, c=z, s=20, cmap="viridis", edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Point Density")
        except Exception:
            ax.scatter(obs, pred, c="navy", alpha=0.4, s=15)
    else:
        ax.scatter(obs, pred, c="navy", alpha=0.3, s=10)

    lo, hi = _axis_limits(obs, pred)
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="1:1 Line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Observed Depth (m)")
    ax.set_ylabel("Predicted Depth (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    text = (
        f"$R^2 = {stats['R2']:.3f}$\n"
        f"$RMSE = {stats['RMSE']:.2f}$ m\n"
        f"$wMAPE = {stats['wMAPE']:.1f}$%"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )


def plot_scatter(obs, p3, p4, s3, s4, out_dir):
    has_p4 = p4 is not None and s4 is not None
    use_kde = len(obs) <= 5000

    if has_p4:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _subplot_scatter(axes[0], obs, p3, s3, "Phase 03: Global Model", use_kde)
        _subplot_scatter(axes[1], obs, p4, s4, "Phase 04: Refined / Best Map", use_kde)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        _subplot_scatter(ax, obs, p3, s3, "Phase 03: Global Model", use_kde)

    plt.suptitle(
        "Observed vs Predicted Depth (Validation Set)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "5_Plot_Scatter_Comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals(obs, p3, p4, out_dir):
    has_p4 = p4 is not None
    res3 = p3 - obs

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        obs, res3, alpha=0.35, label="Phase 03 Residuals", color="gray", s=15, zorder=2
    )
    if has_p4:
        res4 = p4 - obs
        ax.scatter(
            obs,
            res4,
            alpha=0.35,
            label="Phase 04 Residuals",
            color="dodgerblue",
            s=15,
            zorder=3,
        )
    ax.axhline(0, color="red", linestyle="--", lw=2, label="Zero Error")

    if has_p4:
        rmse4 = np.sqrt(np.mean(res4**2))
        ax.axhline(rmse4, color="dodgerblue", linestyle=":", lw=1, alpha=0.7)
        ax.axhline(
            -rmse4,
            color="dodgerblue",
            linestyle=":",
            lw=1,
            alpha=0.7,
            label=f"±RMSE P4 ({rmse4:.2f} m)",
        )

    ax.set_xlabel("Observed Depth (m)")
    ax.set_ylabel("Residual Error  (Predicted − Observed) [m]")
    ax.set_title("Residual Analysis vs Depth (Validation Set)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "5_Plot_Residuals.png"), dpi=150)
    plt.close()


def plot_histograms(obs, p3, p4, out_dir):
    has_p4 = p4 is not None
    res3 = p3 - obs

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        res3,
        color="gray",
        label="Phase 03 Error",
        kde=True,
        stat="density",
        alpha=0.4,
        element="step",
        ax=ax,
    )
    if has_p4:
        res4 = p4 - obs
        sns.histplot(
            res4,
            color="dodgerblue",
            label="Phase 04 Error",
            kde=True,
            stat="density",
            alpha=0.4,
            element="step",
            ax=ax,
        )

    ax.axvline(0, color="red", linestyle="--", lw=2, label="Zero Error")
    ax.axvline(
        np.mean(res3),
        color="gray",
        linestyle=":",
        lw=1.5,
        label=f"Bias P3 ({np.mean(res3):+.2f} m)",
    )
    if has_p4:
        ax.axvline(
            np.mean(res4),
            color="dodgerblue",
            linestyle=":",
            lw=1.5,
            label=f"Bias P4 ({np.mean(res4):+.2f} m)",
        )

    ax.set_title("Error Distribution (Validation Set)")
    ax.set_xlabel("Error  (Predicted − Observed) [m]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "5_Plot_Error_Histogram.png"), dpi=150)
    plt.close()


def run_phase05_reporting(algorithm, parameters, context, feedback):
    out_dir = algorithm.parameterAsString(parameters, algorithm.OUTPUT_FOLDER, context)
    os.makedirs(out_dir, exist_ok=True)

    p3_path = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_MAP_P3, context
    ).source()
    p4_path = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_MAP_P4, context
    ).source()

    train_lyr = algorithm.parameterAsVectorLayer(
        parameters, algorithm.INPUT_TRAIN, context
    )
    train_fld = algorithm.parameterAsString(parameters, algorithm.FIELD_TRAIN, context)
    val_lyr = algorithm.parameterAsVectorLayer(
        parameters, algorithm.INPUT_VALIDATION, context
    )
    val_fld = algorithm.parameterAsString(
        parameters, algorithm.FIELD_VAL_DEPTH, context
    )

    has_p4 = p4_path is not None and str(p4_path).strip() != "" and p4_path != "None" and os.path.exists(p4_path) and p4_path != p3_path

    feedback.pushInfo("\n" + "=" * 60)
    feedback.pushInfo(">>> PHASE 05: SCIENTIFIC VALIDATION & REPORTING")
    feedback.pushInfo(f"    P3 map : {os.path.basename(p3_path)}")
    if has_p4:
        feedback.pushInfo(f"    P4 map : {os.path.basename(p4_path)}")
    else:
        feedback.pushInfo("    P4 map : [Bypassed / Not Generated]")
    feedback.pushInfo("=" * 60)

    feedback.pushInfo("\n  [1/5] Sampling rasters at validation points...")
    y_val, val_p3, val_p4 = extract_values(val_lyr, val_fld, p3_path, p4_path, feedback)

    if len(y_val) < 5:
        raise QgsProcessingException(
            f"Only {len(y_val)} valid validation points found — need at least 5."
        )

    feedback.pushInfo(f"  Valid validation samples: {len(y_val)}")

    feedback.pushInfo("  Sampling rasters at training points (reference)...")
    y_train, train_p3, train_p4 = extract_values(
        train_lyr, train_fld, p3_path, p4_path, feedback
    )
    feedback.pushInfo(f"  Valid training samples: {len(y_train)}")

    feedback.pushInfo("\n  [2/5] Calculating statistics...")
    stats_p3 = calc_stats(y_val, val_p3)
    stats_p4 = calc_stats(y_val, val_p4) if has_p4 else None

    strat_rows = []
    strat_rows.extend(stratified_analysis(y_val, val_p3, "Phase 03 (Global)"))
    if has_p4:
        strat_rows.extend(stratified_analysis(y_val, val_p4, "Phase 04 (Refined)"))
    pd.DataFrame(strat_rows).to_csv(
        os.path.join(out_dir, "5_Stratified_Error_Analysis.csv"), index=False
    )

    feedback.pushInfo("\n  [3/5] Exporting raw prediction data...")
    df_dict = {
        "Set": "Validation",
        "Observed": y_val,
        "P3_Pred": val_p3,
        "P3_Error": val_p3 - y_val,
    }
    if has_p4:
        df_dict["P4_Pred"] = val_p4
        df_dict["P4_Error"] = val_p4 - y_val
    df_val = pd.DataFrame(df_dict)
    df_val.to_csv(os.path.join(out_dir, "5_Validation_Raw_Data.csv"), index=False)

    feedback.pushInfo("\n  [4/5] Generating plots...")
    plot_scatter(y_val, val_p3, val_p4, stats_p3, stats_p4, out_dir)
    plot_residuals(y_val, val_p3, val_p4, out_dir)
    plot_histograms(y_val, val_p3, val_p4, out_dir)

    feedback.pushInfo("\n  [5/5] Writing final summary report...")
    report_path = write_final_verdict(
        out_dir,
        stats_p3,
        stats_p4,
        len(y_val),
        os.path.basename(p3_path),
        os.path.basename(p4_path) if has_p4 else "None",
    )

    feedback.pushInfo(f"\n>>> Phase 05 complete. Reports saved to: {out_dir}")
    return {"OUTPUT_REPORT": report_path}
