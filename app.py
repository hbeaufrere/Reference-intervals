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
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from stats_engine import (
    calculate_reference_interval,
    ReferenceIntervalResult,
)
from interpretation import get_interpretation
from example_data import get_example_dataframe

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


def _render_detail(r: ReferenceIntervalResult, df: pd.DataFrame, limit_conf: float,
                   remove_outliers: bool = True):
    """Render detailed results for a single analyte inside a tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Descriptive Statistics**")
        stats_data = {
            "Statistic": ["n (total)", "Missing values", "n (used)",
                          "Outliers removed",
                          "Mean", "Median", "SD", "Min", "Max"],
            "Value": [
                str(r.n_total), str(r.n_missing), str(r.n_used),
                str(r.n_outliers),
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
    all_values = df[r.analyte].dropna().values

    # Separate clean data from outliers based on the remove_outliers setting
    outliers_removed = remove_outliers and r.n_outliers > 0
    if outliers_removed and r.outlier_indices:
        # Build a mask over the non-NaN values
        mask = np.ones(len(all_values), dtype=bool)
        for idx in r.outlier_indices:
            if idx < len(mask):
                mask[idx] = False
        clean_values = all_values[mask]
        outlier_vals = all_values[~mask]
    else:
        clean_values = all_values
        outlier_vals = np.array([])

    # The values used for plots match what was used for RI calculation
    plot_values = clean_values if outliers_removed else all_values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Histogram with RI overlay
    ax = axes[0]
    ax.hist(plot_values, bins="auto", color="#4a90d9", edgecolor="white",
            alpha=0.8, label="Used data")
    if outliers_removed and len(outlier_vals) > 0:
        ax.hist(outlier_vals, bins="auto", color="#e67e22", edgecolor="white",
                alpha=0.6, label=f"Outliers removed ({len(outlier_vals)})")
    ax.axvline(r.lower_limit, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Lower RI ({_fmt(r.lower_limit, 2)})")
    ax.axvline(r.upper_limit, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Upper RI ({_fmt(r.upper_limit, 2)})")
    if not np.isnan(r.lower_ci_low):
        ax.axvspan(r.lower_ci_low, r.lower_ci_high, alpha=0.15, color="#e74c3c")
        ax.axvspan(r.upper_ci_low, r.upper_ci_high, alpha=0.15, color="#e74c3c")
    ax.set_xlabel(r.analyte)
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram" + (" (outliers removed)" if outliers_removed else ""))
    ax.legend(fontsize=7)

    # Box plot
    ax = axes[1]
    ax.boxplot(plot_values, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4a90d9", alpha=0.6),
               medianprops=dict(color="#e74c3c", linewidth=2))
    if outliers_removed and len(outlier_vals) > 0:
        ax.scatter(
            [1] * len(outlier_vals), outlier_vals,
            color="#e67e22", zorder=5, s=40, marker="x", linewidths=2,
            label=f"Outliers removed ({len(outlier_vals)})",
        )
        ax.legend(fontsize=7)
    ax.set_ylabel(r.analyte)
    ax.set_title("Box Plot" + (" (outliers removed)" if outliers_removed else ""))
    ax.set_xticklabels([r.analyte])

    # Q-Q plot
    ax = axes[2]
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(
        plot_values, dist="norm")
    ax.scatter(osm, osr, s=15, color="#4a90d9", alpha=0.7)
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept, color="#e74c3c", linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot" + (" (outliers removed)" if outliers_removed else ""))

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
                "n_missing": r.n_missing,
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

**Le Boedec adjusted method:**

7. Le Boedec K. Sensitivity and specificity of normality tests and
   consequences on reference interval accuracy at small sample size:
   a computer-simulation study. *Vet Clin Pathol.* 2016;45(4):648-656.
   doi:[10.1111/vcp.12390](https://doi.org/10.1111/vcp.12390)

8. Le Boedec K. Reference interval estimation of small sample sizes:
   a methodologic comparison using a computer-simulation study.
   *Vet Clin Pathol.* 2019;48(2):335-346.
   doi:[10.1111/vcp.12725](https://doi.org/10.1111/vcp.12725)

**Software:**

9. Finnegan D. referenceIntervals: Reference Intervals. R package version
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
    options=["auto", "le_boedec", "nonparametric", "robust", "parametric"],
    index=0,
    format_func={
        "auto": "Automatic (ASVCP guideline)",
        "le_boedec": "Automatic (Le Boedec adjustment)",
        "nonparametric": "Nonparametric",
        "robust": "Robust (Horn / CLSI biweight)",
        "parametric": "Parametric",
    }.get,
    help=(
        "**ASVCP auto:** nonparametric if n >= 120, robust if 20 <= n < 120, "
        "parametric otherwise.\n\n"
        "**Le Boedec:** Uses Shapiro-Wilk with raised P > 0.2 threshold "
        "(instead of 0.05) for small samples. Selects parametric if Gaussian, "
        "nonparametric otherwise. Based on Le Boedec 2016 & 2019."
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

# Resolve API key from secrets or environment
_api_key = ""
try:
    _api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    _api_key = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='text-align: center;'>Reference Interval Calculator</h1>"
    "<p style='text-align: center; font-size: 0.9em; margin-top: -10px;'>"
    "Hugues Beaufrère, DVM, PhD, DACZM<br>"
    "Mélanie Ammersbach, DVM, DACVP<br>"
    "UC Davis \u2013 Weill School of Veterinary Medicine</p>",
    unsafe_allow_html=True,
)
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

st.markdown("**Or** try with a built-in example dataset:")
if st.button("Load example dataset (cockatiel lipid panel)"):
    st.session_state["use_example"] = True
    # Clear previous results when switching datasets
    st.session_state.pop("ri_results", None)
    st.session_state.pop("ri_interpretation", None)

# Determine which data source to use
df = None
if uploaded_file is not None:
    st.session_state.pop("use_example", None)
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()
elif st.session_state.get("use_example"):
    df = get_example_dataframe()
    st.caption(
        "Example dataset: triglycerides and cholesterol (mg/dL) "
        "from 80 mixed-sex cockatiels (*Nymphicus hollandicus*). "
        "Simulated data based on published psittacine lipid ranges."
    )

if df is not None:

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
        all_results: list[ReferenceIntervalResult] = []

        progress = st.progress(0)
        for i, col in enumerate(selected):
            values = df[col].values
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

        # Store results in session state so they persist across reruns
        st.session_state["ri_results"] = all_results
        st.session_state["ri_limit_conf"] = limit_conf
        st.session_state["ri_remove_outliers"] = remove_outliers
        # Clear any previous interpretation when new results are computed
        st.session_state.pop("ri_interpretation", None)

    # ------------------------------------------------------------------
    # Display results (from session state, persists across reruns)
    # ------------------------------------------------------------------
    if "ri_results" in st.session_state:
        all_results = st.session_state["ri_results"]
        stored_limit_conf = st.session_state["ri_limit_conf"]
        stored_remove_outliers = st.session_state["ri_remove_outliers"]

        st.header("3. Results")

        # --------------------------------------------------------------
        # Summary table
        # --------------------------------------------------------------
        st.subheader("Summary Table")
        summary_rows = []
        for r in all_results:
            summary_rows.append({
                "Analyte": r.analyte,
                "n": r.n_used,
                "Missing": r.n_missing,
                "Outliers": r.n_outliers,
                "Method": r.method_ri,
                "Lower RI": _fmt(r.lower_limit),
                "Upper RI": _fmt(r.upper_limit),
                f"Lower {stored_limit_conf*100:.0f}% CI": (
                    f"{_fmt(r.lower_ci_low)} \u2013 {_fmt(r.lower_ci_high)}"
                ),
                f"Upper {stored_limit_conf*100:.0f}% CI": (
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
                _render_detail(r, df, stored_limit_conf, stored_remove_outliers)

        # --------------------------------------------------------------
        # AI Interpretation
        # --------------------------------------------------------------
        st.subheader("AI Interpretation (powered by Claude)")
        if not _api_key:
            st.info(
                "Set ANTHROPIC_API_KEY in Streamlit secrets or as an "
                "environment variable to enable AI-powered interpretation."
            )
        else:
            if st.button("Generate AI Interpretation", type="secondary"):
                with st.spinner("Generating interpretation..."):
                    try:
                        interpretation = get_interpretation(
                            all_results, api_key=_api_key,
                        )
                        st.session_state["ri_interpretation"] = interpretation
                    except Exception as e:
                        st.session_state["ri_interpretation"] = None
                        st.error(f"Interpretation failed: {e}")

            # Display stored interpretation
            if "ri_interpretation" in st.session_state and st.session_state["ri_interpretation"]:
                st.markdown(st.session_state["ri_interpretation"])

        # --------------------------------------------------------------
        # Download results
        # --------------------------------------------------------------
        st.subheader("Download Results")
        excel_buf = _results_to_excel(all_results, summary_df, stored_limit_conf)
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
