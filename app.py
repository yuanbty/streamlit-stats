import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Stats Playground", layout="wide")

# Put the seed in the PAGE (top), not sidebar
seed = st.number_input("Random seed", value=42, step=1)
rng = np.random.default_rng(int(seed))

tab_beta, tab_tft = st.tabs(["🏐 Beta + Volleyball", "🎮 TFT Simulator"])

# =========================
# TAB 1: Beta + Volleyball
# =========================
with tab_beta:
    st.title("Beta Distribution Playground + Volleyball Serve Example")
    st.caption("Beta as uncertainty over a probability, then a real-world update using serve outcomes.")

    # --- Beta basics controls (on the page) ---
    st.header("1) Beta Distribution Basics")
    st.write(
    "The **Beta distribution** is a distribution over probabilities (values between 0 and 1). "
    "It’s commonly used to represent uncertainty about a rate like a conversion rate—or a serve success rate."
)

    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.slider("α (alpha)", 0.5, 50.0, 2.0, 0.5, key="beta_a")
    with c2:
        b = st.slider("β (beta)", 0.5, 50.0, 5.0, 0.5, key="beta_b")
    with c3:
        n_samp = st.slider("Samples to draw", 200, 20000, 2000, 200, key="beta_nsamp")

    xs = np.linspace(0.0001, 0.9999, 600)
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
        ax.hist(samp, bins=50, alpha=0.85, density=True)
        ax.set_title("Samples from Beta")
        ax.set_xlabel("p")
        ax.set_ylabel("Density")
        st.pyplot(fig)

    st.markdown("---")

    # --- Volleyball example controls (on the page) ---
    st.header("2) Real-World Example: Volleyball Serve Success Rate")

    st.image(
        "https://images.pexels.com/photos/29898703/pexels-photo-29898703.jpeg",
        caption="Volleyball serves are important and can determine the result of a game.",
        width=250
    )

    st.write(
    "Some background: I'm a huge volleyball lover. I have been playing since elementry school and still play regularly (2-3 times per week) in New York."
    " In volleyball, serve success rate is an important indicator of a player’s consistency and effectiveness, but it is often estimated from a limited number of attempts during practice or matches. Using a Beta distribution allows us to represent not only the observed success rate, but also the uncertainty around a player’s true serving ability. For example, going 7 out of 10 serves may look similar to going 70 out of 100, but the larger sample provides much greater confidence in the estimate. By updating our belief about a player’s true success probability as more serves are observed, this approach provides a more stable and informative performance estimate than raw percentages alone. This type of probabilistic reasoning can help coaches and players make better decisions about training progress, performance evaluation, and expected outcomes in future games.\n"
    " Suppose a player has an unknown true serve success probability **p**. "
    " Each serve is a success/failure (Bernoulli). After observing outcomes, we update our belief about **p**.\n\n"
    "**Key idea:** Beta is conjugate to Bernoulli/Binomial, so the posterior remains Beta and updates are simple."
)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        attempts = st.number_input("Serve attempts (n)", min_value=1, value=30, step=1, key="vb_attempts")
    with c2:
        successes = st.number_input(
            "Successful serves (s)", min_value=0, max_value=int(attempts),
            value=min(20, int(attempts)), step=1, key="vb_successes"
        )
    with c3:
        prior_strength = st.selectbox("Prior strength", ["Uninformative", "Mild", "Strong"], index=1, key="vb_prior")
    with c4:
        target = st.slider("Target rate (for P(p > target))", 0.0, 1.0, 0.6, 0.05, key="vb_target")

    failures = int(attempts - successes)
    if prior_strength == "Uninformative":
        a0, b0 = 1.0, 1.0
    elif prior_strength == "Mild":
        a0, b0 = 5.0, 5.0
    else:
        a0, b0 = 20.0, 20.0

    a_post = a0 + successes
    b_post = b0 + failures

    post_mean = a_post / (a_post + b_post)
    post_ci_lo, post_ci_hi = stats.beta.ppf([0.025, 0.975], a_post, b_post)
    prob_above_target = 1 - stats.beta.cdf(target, a_post, b_post)

    k1, k2, k3 = st.columns(3)
    k1.metric("Posterior mean E[p]", f"{post_mean:.3f}")
    k2.metric("95% credible interval", f"[{post_ci_lo:.3f}, {post_ci_hi:.3f}]")
    k3.metric(f"P(p > {target:.2f})", f"{prob_above_target:.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, stats.beta.pdf(xs, a0, b0), linewidth=2, label=f"Prior Beta({a0:.0f},{b0:.0f})")
    ax.plot(xs, stats.beta.pdf(xs, a_post, b_post), linewidth=2, label=f"Posterior Beta({a_post:.0f},{b_post:.0f})")
    ax.axvline(target, linestyle="--", linewidth=2, label=f"Target = {target:.2f}")
    ax.set_title("Prior vs Posterior for Serve Success Rate")
    ax.set_xlabel("p (serve success probability)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    st.subheader("How uncertainty shrinks as you observe more serves")
    p_hat = successes / attempts
    n_grid = np.array([10, 20, 30, 50, 75, 100, 150, 200, 300])
    widths = []
    for n0 in n_grid:
        s0 = int(round(p_hat * n0))
        f0 = n0 - s0
        a_tmp = a0 + s0
        b_tmp = b0 + f0
        lo_tmp, hi_tmp = stats.beta.ppf([0.025, 0.975], a_tmp, b_tmp)
        widths.append(hi_tmp - lo_tmp)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_grid, widths, marker="o")
    ax.set_xlabel("Number of observed serves")
    ax.set_ylabel("95% credible interval width")
    ax.set_title("More observations → narrower uncertainty")
    st.pyplot(fig)

# =========================
# TAB 2: TFT
# =========================
with tab_tft:
    st.title("TFT Probability Simulator: 3-Star a 5-Cost at Level 10")

    st.markdown(
"""
**Background (for readers unfamiliar with TFT):**  
Teamfight Tactics (TFT) is an auto-battler where you spend gold to refresh a shop and buy champions.  
Champions come in cost tiers (1–5). Higher-cost champions are rarer in the shop. Buying multiple copies
upgrades a champion; collecting **9 copies** upgrades it to **3-star**. In many TFT sets, a **3-star 5-cost**
is effectively an “instant win” condition because it is extremely powerful and very difficult to obtain.

This is probably the game that I play the most. This games involves lots of probabilities that require you make descisions in every round in order to win.
I consistently climb to Master/GrandMaster each season and l watch a lot of twitch streamers to improve my game knowledge.

This page models the question: **how much gold do you need to have to reasonably hit 9 copies of a specific 5-cost champion, and what is the probability?**
We estimate this via simulation under simplified assumptions.

Be Aware: the last graph may take a bit to run due to large amount of simulation. 

In practice, the probability can be higher in real matches because of the champion pool–sharing mechanism: other players may hold copies of different 5-cost champions, which reduces the total number of available 5-cost units in the pool and slightly increases the chance of finding the specific champion you want. There are also many additional game-mechanic factors that I ignore here to keep the simulation simple and interpretable.

Realistically, players often need less gold than this simplified model suggests to hit a 3-star 5-cost. As a fun fact, in my eight years of playing TFT, I have only managed to achieve a 3-star 5-cost twice in competitive ranked matches, which reflects how rare this outcome truly is (But I did not play that many of games each season to be honest).
"""
)

    st.subheader("My TFT Rank")

    st.image(
    "assets/rank.png",
    caption="Just an screenshot to show I'm not lying.",
    width=350
    )

    # Put TFT controls 
    st.subheader("Controls")

    roll_cost = 2
    buy_cost = 5

    c3, c4 = st.columns(2)
    with c3:
        gold = st.slider("Starting gold", 120, 400, 200, 5, key="tft_gold")
    with c4:
        B = st.slider("Simulations (Monte Carlo)", 200, 10000, 1000, 200, key="tft_B")

    show_curve = st.checkbox("Show probability curve vs gold", value=True, key="tft_curve")
    if show_curve:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            gold_min = st.slider("Curve min gold", 100, 300, 120, 1, key="tft_gmin")
        with cc2:
            gold_max = st.slider("Curve max gold", 200, 800, 400, 1, key="tft_gmax")
        with cc3:
            gold_step = st.slider("Curve step", 1, 20, 5, 1, key="tft_gstep")

    # --- TFT assumptions and sim (paste your existing TFT sim code here) ---
    slots_per_shop = 5
    p_five_cost = 0.25
    n_five_cost_champs = 18
    copies_per_champ = 9
    target_copies = 9
    total_pool_initial = n_five_cost_champs * copies_per_champ

    def simulate_once(start_gold: int) -> dict:
        g = start_gold
        bought = 0
        rolls = 0
        while True:
            if bought >= target_copies:
                return {"success": 1, "bought": bought, "rolls": rolls, "gold_left": g}
            if g < roll_cost:
                return {"success": 0, "bought": bought, "rolls": rolls, "gold_left": g}

            g -= roll_cost
            rolls += 1

            remaining_target = copies_per_champ - bought
            remaining_total = total_pool_initial - bought
            p_target_given_five = remaining_target / remaining_total
            p_target_slot = p_five_cost * p_target_given_five

            for _ in range(slots_per_shop):
                if rng.random() < p_target_slot:
                    if g >= buy_cost:
                        g -= buy_cost
                        bought += 1
                        if bought >= target_copies:
                            return {"success": 1, "bought": bought, "rolls": rolls, "gold_left": g}

    def run_sim(start_gold: int, sims: int) -> pd.DataFrame:
        return pd.DataFrame(simulate_once(start_gold) for _ in range(sims))

    df = run_sim(gold, B)
    p_success = df["success"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated P(3-star)", f"{p_success*100:.1f}%")
    m2.metric("Avg copies bought", f"{df['bought'].mean():.2f} / 9")
    m3.metric("Avg rolls used", f"{df['rolls'].mean():.1f}")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["bought"], bins=np.arange(0, 11) - 0.5, alpha=0.85)
        ax.set_title("Copies obtained")
        ax.set_xlabel("Copies")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["rolls"], bins=40, alpha=0.85)
        ax.set_title("Rolls used")
        ax.set_xlabel("Rolls")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    if show_curve:
        gold_grid = list(range(gold_min, gold_max + 1, gold_step))
        B_curve = min(3000, B)
        probs = []
        for g0 in gold_grid:
            probs.append(run_sim(g0, B_curve)["success"].mean())

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(gold_grid, np.array(probs) * 100, marker="o")
        ax.set_xlabel("Starting gold")
        ax.set_ylabel("Estimated P(3-star) (%)")
        ax.set_title(f"Probability to hit 9 copies vs gold (B={B_curve} per point)")
        st.pyplot(fig)
