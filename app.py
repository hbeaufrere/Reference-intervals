"""
Reference Interval Calculator

A Streamlit application for calculating de novo reference intervals from
Excel data, following the ASVCP guidelines (Friedrichs et al., 2012) and
the CLSI EP28-A3c standard.

Usage
-----
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from stats_engine import (
    calculate_reference_interval,
    ReferenceIntervalResult,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Reference Interval Calculator",
    page_icon="\U0001F4CA",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fmt(val, decimals=4):
    """Format a numeric value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1000:
        return f"{val:,.2f}"
    return f"{val:.{decimals}f}"


def _render_detail(r: ReferenceIntervalResult, df: pd.DataFrame, limit_conf: float):
    """Render detailed results for a single analyte inside a tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Descriptive Statistics**")
        stats_data = {
            "Statistic": ["n (total)", "n (used)", "Outliers removed",
                          "Mean", "Median", "SD", "Min", "Max"],
            "Value": [
                str(r.n_total), str(r.n_used), str(r.n_outliers),
                _fmt(r.mean), _fmt(r.median), _fmt(r.std),
                _fmt(r.min_val), _fmt(r.max_val),
            ],
        }
        st.dataframe(
            pd.DataFrame(stats_data), hide_index=True, use_container_width=True,
        )

        st.markdown("**Normality Assessment**")
        norm_data = {
            "Test": ["Shapiro-Wilk p-value", "Anderson-Darling statistic",
                     "Anderson-Darling critical (5%)", "Normal distribution?"],
            "Value": [
                _fmt(r.shapiro_p), _fmt(r.anderson_stat),
                _fmt(r.anderson_critical),
                "Yes" if r.is_normal else "No" if r.is_normal is not None else "N/A",
            ],
        }
        st.dataframe(
            pd.DataFrame(norm_data), hide_index=True, use_container_width=True,
        )

        if r.boxcox_lambda is not None:
            st.markdown(f"**Box-Cox lambda:** {r.boxcox_lambda:.4f}")

    with col2:
        st.markdown("**Reference Interval**")
        ri_data = {
            "": ["RI Method", "Lower Limit", "Upper Limit",
                 "CI Method",
                 f"Lower Limit {limit_conf*100:.0f}% CI",
                 f"Upper Limit {limit_conf*100:.0f}% CI"],
            "Value": [
                r.method_ri,
                _fmt(r.lower_limit),
                _fmt(r.upper_limit),
                r.method_ci,
                f"{_fmt(r.lower_ci_low)} \u2013 {_fmt(r.lower_ci_high)}",
                f"{_fmt(r.upper_ci_low)} \u2013 {_fmt(r.upper_ci_high)}",
            ],
        }
        st.dataframe(
            pd.DataFrame(ri_data), hide_index=True, use_container_width=True,
        )

        if r.outlier_values:
            st.markdown(
                f"**Outliers detected ({r.n_outliers}):** "
                f"{', '.join(_fmt(v) for v in r.outlier_values)}"
            )

        if r.warnings:
            for w in r.warnings:
                st.warning(w)

    # --- Plots ---
    st.markdown("**Distribution**")
    values = df[r.analyte].dropna().values
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Histogram with RI overlay
    ax = axes[0]
    ax.hist(values, bins="auto", color="#4a90d9", edgecolor="white", alpha=0.8)
    ax.axvline(r.lower_limit, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Lower RI ({_fmt(r.lower_limit, 2)})")
    ax.axvline(r.upper_limit, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Upper RI ({_fmt(r.upper_limit, 2)})")
    if not np.isnan(r.lower_ci_low):
        ax.axvspan(r.lower_ci_low, r.lower_ci_high, alpha=0.15, color="#e74c3c")
        ax.axvspan(r.upper_ci_low, r.upper_ci_high, alpha=0.15, color="#e74c3c")
    ax.set_xlabel(r.analyte)
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram")
    ax.legend(fontsize=7)

    # Box plot
    ax = axes[1]
    ax.boxplot(values, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4a90d9", alpha=0.6),
               medianprops=dict(color="#e74c3c", linewidth=2))
    if r.outlier_values:
        ax.scatter(
            [1] * len(r.outlier_values), r.outlier_values,
            color="#e74c3c", zorder=5, s=40, label="Outliers",
        )
        ax.legend(fontsize=7)
    ax.set_ylabel(r.analyte)
    ax.set_title("Box Plot")
    ax.set_xticklabels([r.analyte])

    # Q-Q plot
    ax = axes[2]
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(values, dist="norm")
    ax.scatter(osm, osr, s=15, color="#4a90d9", alpha=0.7)
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept, color="#e74c3c", linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _results_to_excel(results, summary_df, limit_conf):
    """Export results to an Excel file in memory."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        # Summary sheet
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Detail sheet
        detail_rows = []
        for r in results:
            detail_rows.append({
                "Analyte": r.analyte,
                "n_total": r.n_total,
                "n_outliers": r.n_outliers,
                "n_used": r.n_used,
                "Mean": r.mean,
                "Median": r.median,
                "SD": r.std,
                "Min": r.min_val,
                "Max": r.max_val,
                "RI_Method": r.method_ri,
                "Lower_RI": r.lower_limit,
                "Upper_RI": r.upper_limit,
                "CI_Method": r.method_ci,
                "Lower_CI_low": r.lower_ci_low,
                "Lower_CI_high": r.lower_ci_high,
                "Upper_CI_low": r.upper_ci_low,
                "Upper_CI_high": r.upper_ci_high,
                "Shapiro_Wilk_p": r.shapiro_p,
                "Anderson_Darling_stat": r.anderson_stat,
                "Is_Normal": r.is_normal,
                "BoxCox_Lambda": r.boxcox_lambda,
                "Outlier_Values": (
                    "; ".join(f"{v:.4f}" for v in r.outlier_values)
                    if r.outlier_values else ""
                ),
                "Warnings": "; ".join(r.warnings) if r.warnings else "",
            })
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_excel(writer, sheet_name="Details", index=False)

    buf.seek(0)
    return buf


def _render_references():
    """Render the references and citations section."""
    st.markdown("---")
    st.markdown(
        """
### References & Citations

**Guidelines:**

1. Friedrichs KR, Harr KE, Freeman KP, et al. ASVCP reference interval
   guidelines: determination of de novo reference intervals in veterinary
   species and other related topics. *Vet Clin Pathol.* 2012;41(4):441-453.
   doi:[10.1111/vcp.12006](https://doi.org/10.1111/vcp.12006)

2. CLSI. Defining, Establishing, and Verifying Reference Intervals in the
   Clinical Laboratory; Approved Guideline -- Third Edition. CLSI document
   EP28-A3c. Wayne, PA: Clinical and Laboratory Standards Institute; 2010.

**Algorithms:**

3. Horn PS, Pesce AJ, Copeland BE. A robust approach to reference interval
   estimation and evaluation. *Clin Chem.* 1998;44(3):622-631.
   doi:[10.1093/clinchem/44.3.622](https://doi.org/10.1093/clinchem/44.3.622)

4. Horn PS, Pesce AJ. Reference intervals: an update.
   *Clin Chim Acta.* 2003;334(1-2):5-23.
   doi:[10.1016/S0009-8981(03)00133-5](https://doi.org/10.1016/S0009-8981(03)00133-5)

5. Dixon WJ. Processing data for outliers. *Biometrics.* 1953;9(1):74-89.

6. Reed AH, Henry RJ, Mason WB. Influence of statistical method used on
   the resulting estimate of normal range. *Clin Chem.* 1971;17(4):275-284.

**Software:**

7. Finnegan D. referenceIntervals: Reference Intervals. R package version
   1.3.1. 2024. Available from:
   [https://CRAN.R-project.org/package=referenceIntervals](https://CRAN.R-project.org/package=referenceIntervals).
   *The statistical methods in this application were inspired by and
   cross-referenced against this R package.*
"""
    )


# ---------------------------------------------------------------------------
# Sidebar - settings
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")

st.sidebar.subheader("Reference Interval")
ref_conf = st.sidebar.selectbox(
    "Coverage probability",
    options=[0.90, 0.95, 0.99],
    index=1,
    format_func=lambda x: f"{x*100:.0f}%",
    help="Central proportion of the reference population (typically 95%).",
)
limit_conf = st.sidebar.selectbox(
    "Confidence level for limits",
    options=[0.90, 0.95, 0.99],
    index=0,
    format_func=lambda x: f"{x*100:.0f}%",
    help="Confidence level for the CI around each reference limit (typically 90%).",
)

st.sidebar.subheader("Outlier Detection")
outlier_method = st.sidebar.selectbox(
    "Method",
    options=["horn", "tukey", "dixon", "none"],
    index=0,
    format_func={
        "horn": "Horn (Box-Cox + Tukey)",
        "tukey": "Tukey IQR fences",
        "dixon": "Dixon/Reed D/R",
        "none": "None",
    }.get,
    help=(
        "Horn: Box-Cox transform then Tukey fences (recommended by ASVCP). "
        "Tukey: IQR fences without transformation. "
        "Dixon: D/R ratio test for single extreme values."
    ),
)
remove_outliers = st.sidebar.checkbox(
    "Remove detected outliers", value=True,
    help="Exclude outliers from reference interval calculation.",
)

st.sidebar.subheader("RI Method")
ri_method = st.sidebar.selectbox(
    "Calculation method",
    options=["auto", "nonparametric", "robust", "parametric"],
    index=0,
    format_func={
        "auto": "Automatic (ASVCP guideline)",
        "nonparametric": "Nonparametric",
        "robust": "Robust (Horn / CLSI biweight)",
        "parametric": "Parametric",
    }.get,
    help=(
        "Auto: nonparametric if n >= 120, robust if 20 <= n < 120, "
        "parametric otherwise. Per ASVCP guidelines."
    ),
)

st.sidebar.subheader("CI Method")
ci_method = st.sidebar.selectbox(
    "Confidence interval method",
    options=["auto", "parametric", "nonparametric", "bootstrap"],
    index=0,
    format_func={
        "auto": "Automatic",
        "parametric": "Parametric",
        "nonparametric": "Nonparametric (rank-based)",
        "bootstrap": "Bootstrap",
    }.get,
    help=(
        "Auto: matches CI method to RI method. "
        "Bootstrap uses 5000 resamples."
    ),
)

n_boot = st.sidebar.number_input(
    "Bootstrap resamples",
    min_value=1000, max_value=50000, value=5000, step=1000,
    help="Number of bootstrap resamples for CI estimation.",
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Reference Interval Calculator")
st.markdown(
    "Calculate de novo reference intervals following "
    "**ASVCP guidelines** (Friedrichs et al., 2012) and "
    "**CLSI EP28-A3c**."
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

st.header("1. Upload Data")
st.markdown(
    "Upload an Excel file (`.xlsx` / `.xls`). The first column should contain "
    "**animal IDs** and subsequent columns should contain the **analyte values**. "
    "The first row must be **headers**."
)

uploaded_file = st.file_uploader(
    "Choose an Excel file",
    type=["xlsx", "xls"],
    help="Accepted formats: .xlsx, .xls",
)

if uploaded_file is not None:
    # ------------------------------------------------------------------
    # Read and preview data
    # ------------------------------------------------------------------
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    id_col = df.columns[0]
    analyte_cols = df.columns[1:]
    numeric_cols = [c for c in analyte_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) == 0:
        st.error(
            "No numeric analyte columns found. Make sure your data has numeric "
            "values in columns after the ID column."
        )
        st.stop()

    non_numeric = [c for c in analyte_cols if c not in numeric_cols]
    if non_numeric:
        st.warning(
            f"Skipping non-numeric columns: {', '.join(str(c) for c in non_numeric)}"
        )

    st.info(
        f"**{len(df)}** animals | **{len(numeric_cols)}** analytes | "
        f"ID column: **{id_col}**"
    )

    # ------------------------------------------------------------------
    # Select analytes
    # ------------------------------------------------------------------
    st.header("2. Select Analytes")
    selected = st.multiselect(
        "Choose analytes to analyse",
        options=numeric_cols,
        default=list(numeric_cols),
    )

    if not selected:
        st.warning("Select at least one analyte to proceed.")
        st.stop()

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------
    if st.button("Calculate Reference Intervals", type="primary"):
        st.header("3. Results")

        all_results: list[ReferenceIntervalResult] = []

        progress = st.progress(0)
        for i, col in enumerate(selected):
            values = df[col].dropna().values
            result = calculate_reference_interval(
                values,
                analyte_name=str(col),
                outlier_method=outlier_method,
                remove_outliers=remove_outliers,
                ri_method=ri_method,
                ci_method=ci_method,
                ref_conf=ref_conf,
                limit_conf=limit_conf,
                n_boot=n_boot,
            )
            all_results.append(result)
            progress.progress((i + 1) / len(selected))

        progress.empty()

        # --------------------------------------------------------------
        # Summary table
        # --------------------------------------------------------------
        st.subheader("Summary Table")
        summary_rows = []
        for r in all_results:
            summary_rows.append({
                "Analyte": r.analyte,
                "n": r.n_used,
                "Outliers": r.n_outliers,
                "Method": r.method_ri,
                "Lower RI": _fmt(r.lower_limit),
                "Upper RI": _fmt(r.upper_limit),
                f"Lower {limit_conf*100:.0f}% CI": (
                    f"{_fmt(r.lower_ci_low)} \u2013 {_fmt(r.lower_ci_high)}"
                ),
                f"Upper {limit_conf*100:.0f}% CI": (
                    f"{_fmt(r.upper_ci_low)} \u2013 {_fmt(r.upper_ci_high)}"
                ),
                "Normal (p)": (
                    f"{'Yes' if r.is_normal else 'No'} "
                    f"({_fmt(r.shapiro_p)})"
                    if r.is_normal is not None else "N/A"
                ),
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # --------------------------------------------------------------
        # Per-analyte detail tabs
        # --------------------------------------------------------------
        st.subheader("Detailed Results")
        tabs = st.tabs([r.analyte for r in all_results])

        for tab, r in zip(tabs, all_results):
            with tab:
                _render_detail(r, df, limit_conf)

        # --------------------------------------------------------------
        # Download results
        # --------------------------------------------------------------
        st.subheader("Download Results")
        excel_buf = _results_to_excel(all_results, summary_df, limit_conf)
        st.download_button(
            label="Download Results as Excel",
            data=excel_buf,
            file_name="reference_intervals.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ---------------------------------------------------------------------------
# References (always visible)
# ---------------------------------------------------------------------------

_render_references()
