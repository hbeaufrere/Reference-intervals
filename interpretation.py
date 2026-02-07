"""
AI-powered interpretation of reference interval results using the Claude API.

Provides two-section analysis:
1. Quality assessment -- flags issues that may make results unsuitable for
   clinical practice (negative limits, small sample sizes, wide CIs, etc.).
2. General interpretation -- contextual commentary on the reference intervals.
"""

import anthropic
import numpy as np
from stats_engine import ReferenceIntervalResult


def _build_results_summary(results: list[ReferenceIntervalResult]) -> str:
    """Format all analyte results into a structured text block for the prompt."""
    lines = []
    for r in results:
        lines.append(f"Analyte: {r.analyte}")
        lines.append(f"  Sample size: n_total={r.n_total}, n_used={r.n_used}, "
                      f"outliers_removed={r.n_outliers}")
        lines.append(f"  Descriptive: mean={r.mean:.4f}, median={r.median:.4f}, "
                      f"SD={r.std:.4f}, min={r.min_val:.4f}, max={r.max_val:.4f}")
        lines.append(f"  Normality: Shapiro-Wilk p={r.shapiro_p:.4f}, "
                      f"normal={r.is_normal}")
        if r.boxcox_lambda is not None:
            lines.append(f"  Box-Cox lambda: {r.boxcox_lambda:.4f}")
        lines.append(f"  RI method: {r.method_ri}")
        lines.append(f"  Reference interval: {r.lower_limit:.4f} -- "
                      f"{r.upper_limit:.4f}")
        lines.append(f"  CI method: {r.method_ci}")
        lines.append(f"  Lower limit {r.limit_conf*100:.0f}% CI: "
                      f"{r.lower_ci_low:.4f} -- {r.lower_ci_high:.4f}")
        lines.append(f"  Upper limit {r.limit_conf*100:.0f}% CI: "
                      f"{r.upper_ci_low:.4f} -- {r.upper_ci_high:.4f}")
        if r.outlier_values:
            lines.append(f"  Outlier values: "
                          f"{', '.join(f'{v:.4f}' for v in r.outlier_values)}")
        if r.warnings:
            lines.append(f"  Warnings: {'; '.join(r.warnings)}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a veterinary clinical pathologist with deep expertise in reference \
interval methodology. You are well-versed in:
- The **ASVCP guidelines** (Friedrichs et al., Vet Clin Pathol 2012;41:441-453) \
for determination of de novo reference intervals in veterinary species.
- The **CLSI EP28-A3c** standard for defining, establishing, and verifying \
reference intervals.
- **Le Boedec (2016)** (Vet Clin Pathol 2016;45:648-656): demonstrated that \
normality tests at alpha=0.05 have poor specificity (~50%) at small sample \
sizes, recommending a raised Shapiro-Wilk threshold of P > 0.2 to reduce \
erroneous application of parametric methods to non-Gaussian data.
- **Le Boedec (2019)** (Vet Clin Pathol 2019;48:335-346): optimal strategy \
selection for small samples -- use parametric if Shapiro-Wilk P > 0.2, \
nonparametric otherwise; nonparametric for both limits when n < 40; \
standard nonparametric when n >= 120.

You are reviewing reference interval results computed from a dataset \
uploaded by a colleague.

Respond in **Markdown** format with exactly two sections:

## 1. Quality Assessment

Evaluate whether these reference intervals are suitable for clinical use. \
Check for and flag any of the following issues:
- **Negative lower limits** for analytes that are biologically impossible to \
be negative (e.g., concentrations, enzyme activities, cell counts).
- **Sample size concerns** -- n < 40 is genuinely problematic and should \
be flagged clearly. Sample sizes between 40 and 120 are generally \
acceptable; the nonparametric method requires n >= 120, but robust or \
parametric approaches are statistically valid alternatives for smaller \
samples, so do NOT flag 40 <= n < 120 as a major concern.
- **Wide confidence intervals** relative to the RI width, suggesting \
imprecise limits (CI width > 0.2 * RI width is the Harris-Boyd criterion).
- **High proportion of outliers** removed (> 5% may indicate a \
heterogeneous population or pre-analytical issues).
- **Non-normal distributions** where parametric methods were forced.
- **Any other statistical red flags** (e.g., very large SD relative to \
mean, extreme skewness suggested by mean vs median divergence).

For each issue found, explain why it matters clinically and suggest a \
practical remedy. If everything looks good, say so. When evaluating method \
selection, consider whether the Le Boedec adjusted strategy or the \
standard ASVCP approach was used, and whether the choice was appropriate \
for the sample size and distribution.

## 2. General Interpretation

Provide a concise overall interpretation of the reference intervals:
- Comment on the methods selected and whether they are appropriate given \
the sample size and data distribution. Reference the ASVCP guidelines and \
Le Boedec recommendations where relevant.
- Note any analytes where the RI seems unusually wide or narrow.
- Highlight any analytes that may need special attention when used in \
clinical practice.
- Comment on the overall data quality.
- Keep this section practical and useful for a clinical audience.

Be concise but thorough. Do not repeat raw numbers unnecessarily -- the \
user can see the tables. Focus on clinical insight.\
"""


def get_interpretation(
    results: list[ReferenceIntervalResult],
    api_key: str,
    model: str = "claude-sonnet-4-5-20250929",
) -> str:
    """Call the Claude API to interpret reference interval results.

    Parameters
    ----------
    results : list[ReferenceIntervalResult]
        Computed reference interval results for all analytes.
    api_key : str
        Anthropic API key.
    model : str
        Claude model to use.

    Returns
    -------
    str
        Markdown-formatted interpretation text.
    """
    client = anthropic.Anthropic(api_key=api_key)
    summary = _build_results_summary(results)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Please review the following reference interval results "
                    "and provide your interpretation.\n\n"
                    f"{summary}"
                ),
            }
        ],
    )

    # Extract text from the response
    return message.content[0].text
