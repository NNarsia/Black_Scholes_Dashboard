import numpy as np
import scipy.stats as si
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Black-Scholes Pricing Functions
# -------------------------------
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r*T) * si.norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

# -------------------------------
# Greeks
# -------------------------------
def option_greeks(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf = si.norm.pdf(d1)

    if option == "call":
        Delta = si.norm.cdf(d1)
        Theta = -(S * pdf * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * si.norm.cdf(d2)
        Rho = K * T * np.exp(-r*T) * si.norm.cdf(d2)
    else:
        Delta = si.norm.cdf(d1) - 1
        Theta = -(S * pdf * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * si.norm.cdf(-d2)
        Rho = -K * T * np.exp(-r*T) * si.norm.cdf(-d2)

    Gamma = pdf / (S * sigma * np.sqrt(T))
    Vega = S * pdf * np.sqrt(T)

    return Delta, Gamma, Vega, Theta, Rho

# -------------------------------
# Heatmap
# -------------------------------
def plot_pnl_heatmap(K, T, r, purchase_price, option="call"):
    S_vals = np.linspace(50, 150, 8)
    sigma_vals = np.linspace(0.1, 0.6, 8)
    pnl = np.zeros((len(S_vals), len(sigma_vals)))

    for i, S in enumerate(S_vals):
        for j, sig in enumerate(sigma_vals):
            if option == "call":
                price = black_scholes_call(S, K, T, r, sig)
            else:
                price = black_scholes_put(S, K, T, r, sig)
            pnl[i, j] = price - purchase_price

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pnl,
        xticklabels=np.round(sigma_vals, 2),
        yticklabels=np.round(S_vals, 2),
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        ax=ax
    )
    ax.set_xlabel("Volatility σ")
    ax.set_ylabel("Stock Price S")
    ax.set_title(f"{option.capitalize()} Option PnL Heatmap")
    st.pyplot(fig)

# -------------------------------
# Greek Curves (Dark Theme)
# -------------------------------
def plot_greek_curve(K, T, r, sigma, greek, option="call"):
    S_vals = np.linspace(50, 150, 100)
    values = []

    for S in S_vals:
        Delta, Gamma, Vega, Theta, Rho = option_greeks(S, K, T, r, sigma, option)
        if greek == "Delta":
            values.append(Delta)
        elif greek == "Gamma":
            values.append(Gamma)
        elif greek == "Vega":
            values.append(Vega)
        elif greek == "Theta":
            values.append(Theta)
        elif greek == "Rho":
            values.append(Rho)

    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.plot(S_vals, values, color="cyan", linewidth=2, label=greek)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#0E1117")  # Streamlit dark bg
    ax.set_xlabel("Stock Price S")
    ax.set_ylabel(greek)
    ax.set_title(f"{greek} vs Stock Price ({option.capitalize()} Option)")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# Streamlit App Layout
# -------------------------------
# Sidebar header and info
st.sidebar.markdown(
    """
    ## Built by [Nikoloz Narsia](https://www.linkedin.com/in/nikoloz-narsia-a2275b274/)  
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20">
    """,
    unsafe_allow_html=True
)

# Sidebar for user inputs
st.sidebar.header("Option Parameters")
S = st.sidebar.slider("Stock Price S₀", 50, 200, 100)
K = st.sidebar.slider("Strike Price K", 50, 200, 100)
T = st.sidebar.slider("Time to Maturity (years)", 0.1, 5.0, 1.0)
r = st.sidebar.slider("Risk-Free Rate r", 0.0, 0.1, 0.05)
sigma = st.sidebar.slider("Volatility σ", 0.1, 1.0, 0.2)
purchase_price = st.sidebar.number_input("Purchase Price of Option", value=10.0)
option_type = st.sidebar.radio("Option Type", ("Call", "Put"))

# GitHub button 
st.link_button("🌐 View on GitHub", "https://github.com/NNarsia")

# Title
st.title("📈 Black-Scholes Option Pricing Dashboard")

# Intro box
st.info("This dashboard uses the Black–Scholes model to calculate European call and put option prices, "
        "visualize sensitivities (Greeks), and show profit/loss scenarios under different market conditions.")

# Prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)
col1, col2 = st.columns(2)
col1.metric("Call Option Price", f"${call_price:.2f}")
col2.metric("Put Option Price", f"${put_price:.2f}")

# Greeks
Delta, Gamma, Vega, Theta, Rho = option_greeks(S, K, T, r, sigma, option=option_type)
st.subheader(f"Greeks ({option_type.capitalize()} Option)")
st.write(f"Delta: {Delta:.3f} | Gamma: {Gamma:.3f} | Vega: {Vega:.3f} | Theta: {Theta:.3f} | Rho: {Rho:.3f}")

# Greek Curve
greek_choice = st.selectbox("Select Greek to Plot", ["Delta", "Gamma", "Vega", "Theta", "Rho"])
plot_greek_curve(K, T, r, sigma, greek_choice, option=option_type)

# Heatmap
st.subheader("PnL Heatmap")
plot_pnl_heatmap(K, T, r, purchase_price, option=option_type)






