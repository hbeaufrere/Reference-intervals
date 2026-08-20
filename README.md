# Reference Interval Calculator

A Streamlit application for calculating de novo reference intervals from
Excel data, following the ASVCP guidelines (Friedrichs et al., 2012) and
the CLSI EP28-A3c standard. Includes outlier detection, distribution
testing, parametric/nonparametric/robust interval methods, partition
testing, and optional AI-powered interpretation of results.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional: to enable the AI interpretation feature, set an Anthropic API
key either as an environment variable (`ANTHROPIC_API_KEY`) or in
`.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`).
To restrict who can trigger the (paid) AI interpretation calls, also set
`AI_PASSWORD` — users must then enter that password in the app before
the feature unlocks. When `AI_PASSWORD` is unset, the feature is open to
anyone with access to the app.

## Deploy to Posit Connect Cloud

The app deploys to [Posit Connect Cloud](https://connect.posit.cloud)
directly from this GitHub repository — no code changes required.

1. Sign in at [connect.posit.cloud](https://connect.posit.cloud) with
   your GitHub account.
2. Click **Publish** (the `+` button) and choose **Streamlit**.
3. Authorize access to GitHub and select this repository, the branch to
   deploy, and set **`app.py`** as the primary file.
4. Under **Advanced settings**:
   - Choose a Python version (the app is tested on Python 3.11).
   - Add an environment variable `ANTHROPIC_API_KEY` with your Anthropic
     API key to enable the AI interpretation feature (optional —
     everything else works without it). Never commit the key to the repo.
   - Optionally add `AI_PASSWORD` to password-protect the AI
     interpretation feature, so only people you share the password with
     can trigger paid API calls.
5. Click **Publish**. Connect Cloud installs `requirements.txt` and
   builds the app; the first build takes a few minutes.

Subsequent updates: pushes to the deployed branch are picked up
automatically when **auto-publish on push** is enabled in the content
settings; otherwise use the **Republish** button.

## Tests

```bash
python -m pytest
```

Covers the statistical engine (interval methods, outlier detection,
distribution testing, and partition testing) — 59 tests.
