import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Stats Playground", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def rng_from_seed(seed: int):
    return np.random.default_rng(int(seed))

def make_hist(ax, x, bins=40, title="", xlabel="", ylabel="Count"):
    ax.hist(x, bins=bins, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, x)))

# ---------------------------
# Sidebar (global)
# ---------------------------
st.title("Stats Playground: CLT, Confidence Interval Coverage, and Beta Distributions")
st.caption("Interactive simulations and visual intuition for core statistical concepts.")

with st.sidebar:
    st.header("Global")
    seed = st.number_input("Random seed", value=42, step=1)
    rng = rng_from_seed(seed)

tabs = st.tabs(["1) Central Limit Theorem (CLT)", "2) CI Coverage", "3) Beta Distribution"])

# ==========================================================
# TAB 1: CLT
# ==========================================================
with tabs[0]:
    st.subheader("Central Limit Theorem (CLT)")
    st.write(
        "Simulate sample means from a chosen distribution. As sample size *n* increases, "
        "the distribution of sample means becomes approximately normal."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        dist_name = st.selectbox("Base distribution", ["Uniform(0,1)", "Exponential(1)", "Bernoulli(p)"])
    with c2:
        n = st.slider("Sample size n", min_value=1, max_value=5000, value=50, step=1)
    with c3:
        B = st.slider("Simulations B", min_value=200, max_value=10000, value=2000, step=200)

    p = None
    if dist_name.startswith("Bernoulli"):
        p = st.slider("Bernoulli p", min_value=0.05, max_value=0.95, value=0.3, step=0.05)

    # Generate base samples and sample means
    if dist_name == "Uniform(0,1)":
        samples = rng.uniform(0, 1, size=(B, n))
        true_mean, true_var = 0.5, 1/12
        base_label = "Uniform(0,1)"
    elif dist_name == "Exponential(1)":
        samples = rng.exponential(scale=1.0, size=(B, n))
        true_mean, true_var = 1.0, 1.0
        base_label = "Exponential(rate=1)"
    else:
        samples = rng.binomial(n=1, p=p, size=(B, n))
        true_mean, true_var = p, p * (1 - p)
        base_label = f"Bernoulli(p={p:.2f})"

    xbar = samples.mean(axis=1)

    # Theoretical normal approx for sample mean:
    # mean = mu, sd = sqrt(var/n)
    approx_mu = true_mean
    approx_sd = np.sqrt(true_var / n)

    left, right = st.columns(2)

    with left:
        st.markdown("**Base distribution (one sample of size n)**")
        one_sample = samples[0, :]
        fig, ax = plt.subplots(figsize=(7, 4))
        make_hist(ax, one_sample, bins=40, title=f"One sample from {base_label}", xlabel="Value")
        st.pyplot(fig)

    with right:
        st.markdown("**Sampling distribution of the mean**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(xbar, bins=50, alpha=0.75, density=True, label="Simulated sample means")

        # Overlay normal approximation curve
        xs = np.linspace(xbar.min(), xbar.max(), 300)
        ax.plot(xs, stats.norm.pdf(xs, loc=approx_mu, scale=approx_sd), linewidth=2, label="Normal approx")
        ax.set_title("Sample means across simulations")
        ax.set_xlabel("Sample mean")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

    st.markdown("**Quick readout**")
    out = pd.DataFrame({
        "Quantity": ["True mean", "Simulated mean of x̄", "True SD of x̄", "Simulated SD of x̄"],
        "Value": [true_mean, float(xbar.mean()), approx_sd, float(xbar.std(ddof=1))]
    })
    st.table(out)

# ==========================================================
# TAB 2: CI Coverage
# ==========================================================
with tabs[1]:
    st.subheader("Confidence Interval Coverage")
    st.write(
        "Simulate repeated datasets and compute confidence intervals for the mean. "
        "Coverage is the fraction of intervals that contain the true mean."
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        ci_dist = st.selectbox("Data distribution", ["Normal(μ,σ)", "Exponential(1)", "Uniform(0,1)"])
    with c2:
        n_ci = st.slider("Sample size n", min_value=5, max_value=5000, value=50, step=5)
    with c3:
        B_ci = st.slider("Simulations B", min_value=200, max_value=20000, value=2000, step=200)
    with c4:
        conf = st.selectbox("Confidence level", ["90%", "95%", "99%"], index=1)

    alpha_level = {"90%": 0.10, "95%": 0.05, "99%": 0.01}[conf]
    z = stats.norm.ppf(1 - alpha_level/2)

    mu, sigma = None, None
    if ci_dist == "Normal(μ,σ)":
        mu = st.slider("μ (true mean)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        sigma = st.slider("σ (true SD)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        true_mu = mu
        samples = rng.normal(loc=mu, scale=sigma, size=(B_ci, n_ci))
        # CI uses known sigma assumption? We'll do classic large-sample z with sample sd (practical).
    elif ci_dist == "Exponential(1)":
        true_mu = 1.0
        samples = rng.exponential(scale=1.0, size=(B_ci, n_ci))
    else:
        true_mu = 0.5
        samples = rng.uniform(0, 1, size=(B_ci, n_ci))

    xbar = samples.mean(axis=1)
    s = samples.std(axis=1, ddof=1)
    se = s / np.sqrt(n_ci)

    lo = xbar - z * se
    hi = xbar + z * se

    covered = (lo <= true_mu) & (true_mu <= hi)
    coverage = covered.mean()
    avg_len = (hi - lo).mean()

    k_show = clamp_int(min(200, B_ci), 50, 200)  # show first 50-200 intervals

    st.markdown("**Results**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated coverage", f"{coverage*100:.1f}%")
    m2.metric("Avg interval length", f"{avg_len:.4f}")
    m3.metric("Target level", conf)

    # Plot intervals (like classic coverage plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(k_show)
    ax.hlines(idx, lo[:k_show], hi[:k_show], alpha=0.8)
    ax.vlines(true_mu, -1, k_show, linestyles="dashed", linewidth=2)
    ax.set_title(f"First {k_show} confidence intervals for the mean (dashed line = true mean)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Interval index")
    st.pyplot(fig)

    st.markdown(
        "Interpretation tip: If the procedure is calibrated, coverage should be close to the stated confidence level "
        "over repeated samples. Small n or non-normal data can cause deviations."
    )

# ==========================================================
# TAB 3: Beta Distribution
# ==========================================================
with tabs[2]:
    st.subheader("Beta Distribution (probability uncertainty)")
    st.write(
        "The Beta distribution models uncertainty over a probability value (between 0 and 1). "
        "It’s widely used for Bayesian reasoning about conversion rates and proportions."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        a = st.slider("α (alpha)", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
    with c2:
        b = st.slider("β (beta)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
    with c3:
        n_samp = st.slider("Samples to draw", min_value=200, max_value=20000, value=2000, step=200)

    # Summary stats
    mean = a / (a + b)
    var = (a * b) / (((a + b) ** 2) * (a + b + 1))
    mode = None
    if a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)

    # Plot PDF and samples
    xs = np.linspace(0.0001, 0.9999, 500)
    pdf = stats.beta.pdf(xs, a, b)
    samp = rng.beta(a, b, size=n_samp)

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, pdf, linewidth=2)
        ax.set_title("Beta PDF")
        ax.set_xlabel("p")
        ax.set_ylabel("Density")
        st.pyplot(fig)

    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(samp, bins=50, alpha=0.8, density=True)
        ax.set_title("Samples from Beta")
        ax.set_xlabel("p")
        ax.set_ylabel("Density")
        st.pyplot(fig)

    st.markdown("**Key quantities**")
    cA, cB, cC = st.columns(3)
    cA.metric("Mean", f"{mean:.3f}")
    cB.metric("Variance", f"{var:.5f}")
    if mode is None:
        cC.metric("Mode", "N/A (α≤1 or β≤1)")
    else:
        cC.metric("Mode", f"{mode:.3f}")

    st.markdown(
        "Business framing: Beta(α,β) can represent a belief about a conversion rate. "
        "Larger α+β means more certainty; α/(α+β) is the expected rate."
    )
