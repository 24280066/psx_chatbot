import streamlit as st
import pandas as pd
from openai import OpenAI

# 👇 Update this if your CSV name is different
CSV_FILE = "psx_stock_cards_ml.csv"  # must contain at least 'ticker' and 'stock_card'


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE)
    if "ticker" not in df.columns or "stock_card" not in df.columns:
        st.error("CSV must contain at least 'ticker' and 'stock_card' columns.")
    return df


def compute_risk_level(row):
    """
    Simple risk score based on volatility + liquidity.
    This is just heuristic for UI.
    """
    vol = row.get("vol_annual", None)
    liq = str(row.get("liquidity", "")).lower()

    level = "Unknown"
    score = 0.5

    if vol is not None and pd.notna(vol):
        if vol < 0.15:
            level = "Low"
            score = 0.25
        elif vol < 0.25:
            level = "Medium"
            score = 0.5
        else:
            level = "High"
            score = 0.8

    # Liquidity adjustment
    if "low" in liq:
        level = "Higher (due to low liquidity)"
        score = min(1.0, score + 0.15)
    elif "high" in liq and score < 0.7:
        level = "Moderate (with good liquidity)"
        score = max(0.35, score - 0.1)

    return level, score


def find_similar_stocks(df, base_row, top_n=5):
    """
    Simple numeric similarity based on returns, volatility, score.
    No AI here – just distance on standardized features.
    """
    similarity_features = [
        col
        for col in ["return_1y", "return_3m", "return_1m", "vol_annual", "score"]
        if col in df.columns
    ]
    if not similarity_features:
        return pd.DataFrame()

    numeric_df = df[similarity_features].copy()

    # Standardize columns
    for col in similarity_features:
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        if std == 0 or pd.isna(std):
            numeric_df[col] = 0
        else:
            numeric_df[col] = (numeric_df[col] - mean) / std

    base_vec = numeric_df.loc[base_row.name].values

    # Squared distance
    dists = ((numeric_df - base_vec) ** 2).sum(axis=1)

    # Penalty if liquidity is very different
    if "liquidity" in df.columns:
        base_liq = str(base_row.get("liquidity", ""))
        penalty = df["liquidity"].apply(
            lambda x: 0 if str(x) == base_liq else 0.5
        )
        dists = dists + penalty

    similar_idx = dists.sort_values().index
    similar_idx = [i for i in similar_idx if i != base_row.name]  # exclude self

    top_idx = similar_idx[:top_n]

    cols_to_show = [
        c
        for c in [
            "ticker",
            "close",
            "return_1y",
            "return_3m",
            "return_1m",
            "vol_annual",
            "liquidity",
            "score",
        ]
        if c in df.columns
    ]

    return df.loc[top_idx, cols_to_show]


def build_search_tip(df: pd.DataFrame) -> str:
    """
    Build a dynamic search tip based on columns that exist in the data.
    """
    suggestions = []

    # Liquidity-related hints
    if "liquidity" in df.columns:
        suggestions += ["low liquidity", "medium liquidity", "high liquidity"]

    # Volatility
    if "vol_annual" in df.columns:
        suggestions.append("volatility")

    # Return horizons
    if "return_1y" in df.columns:
        suggestions.append("1-year return")
    if "return_3m" in df.columns:
        suggestions.append("3-month return")
    if "return_1m" in df.columns:
        suggestions.append("1-month return")

    # ML signal hints
    if "ml_signal_1m" in df.columns:
        suggestions += ["model_leans_up", "model_leans_down", "uncertain"]

    # De-duplicate while preserving order
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    if not unique_suggestions:
        return (
            "Tip: you can search by ticker symbol or any word that appears "
            "in the stock card."
        )

    top = unique_suggestions[:6]
    examples = ", ".join(f"'{s}'" for s in top)
    return f"Tip: try searching {examples}, or any ticker symbol."


# ---------- STREAMLIT APP ----------

st.set_page_config(
    page_title="PSX Stock Advisor Chatbot",
    page_icon="📈",
    layout="wide",
)

df = load_data()
search_tip = build_search_tip(df)

st.title("📈 PSX Stock Advisor Chatbot")
st.write(
    """
This assistant answers questions about **KSE-100 stocks** using your pre-computed
stock cards (which already include ML predictions, volatility, liquidity, and scores).

It does **not** fetch real-time data and does **not** give investment advice.
"""
)

# --- Sidebar: API key + clear chat + optional sector filter ---

st.sidebar.header("🔐 API & Controls")

api_key = st.sidebar.text_input("OpenAI API key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

if not api_key:
    st.sidebar.info("Enter your OpenAI API key to enable answers.")

# Clear chat button
if st.sidebar.button("🧹 Clear conversation"):
    st.session_state["messages"] = []
    st.sidebar.success("Chat cleared.")

st.sidebar.markdown("---")

# Sector filter (if you later add a 'sector' column)
sector_filter = None


# ---------- MAIN LAYOUT: two columns ----------

left_col, right_col = st.columns([1.1, 1.4])

# ========== LEFT: ticker, risk, ML, chart, similar stocks ==========
with left_col:
    st.markdown("### 1️⃣ Choose a stock")

    # Apply sector filter if available
    if sector_filter and sector_filter != "All" and "sector" in df.columns:
        filtered_df = df[df["sector"] == sector_filter].copy()
    else:
        filtered_df = df.copy()

    all_tickers = sorted(filtered_df["ticker"].unique())
    selected_ticker = st.selectbox("Select PSX ticker:", all_tickers)

    stock_row = filtered_df[filtered_df["ticker"] == selected_ticker].iloc[0]

    st.markdown(f"**You selected:** `{selected_ticker}`")
    st.markdown("---")

    # Risk meter
    st.markdown("#### 🧮 Risk snapshot")
    risk_level, risk_score = compute_risk_level(stock_row)
    st.write(f"**Risk level:** {risk_level}")
    st.progress(min(1.0, max(0.0, risk_score)))

    st.markdown("---")

    # ML outlook section
    st.markdown("#### 🤖 ML 1-month outlook")

    ml_prob = (
        stock_row.get("ml_prob_up_1m")
        if "ml_prob_up_1m" in stock_row.index
        else None
    )
    ml_signal = (
        stock_row.get("ml_signal_1m")
        if "ml_signal_1m" in stock_row.index
        else None
    )

    if ml_prob is not None and pd.notna(ml_prob):
        st.write(
            f"**Probability of price being higher after ~1 month:** "
            f"{ml_prob * 100:.1f}%"
        )
    else:
        st.write("ML probability: not available in this dataset.")

    if isinstance(ml_signal, str):
        st.write(f"**Model signal:** {ml_signal}")
    else:
        st.write("Model signal: not available in this dataset.")

    st.caption(
        "This ML view comes from your pre-computed Random Forest model and is "
        "for educational purposes only."
    )

    st.markdown("---")

    # Performance snapshot chart
    st.markdown("#### 📊 Performance snapshot")

    perf_data = {}
    if "return_1m" in stock_row.index:
        perf_data["1M"] = stock_row["return_1m"]
    if "return_3m" in stock_row.index:
        perf_data["3M"] = stock_row["return_3m"]
    if "return_1y" in stock_row.index:
        perf_data["1Y"] = stock_row["return_1y"]

    if perf_data:
        # Build DF and enforce x-axis order: 1M, 3M, 1Y
        perf_df = pd.DataFrame(
            {"Period": list(perf_data.keys()), "Return": list(perf_data.values())}
        ).set_index("Period")

        order = ["1M", "3M", "1Y"]
        perf_df = perf_df.reindex(order).dropna()

        st.bar_chart(perf_df)
        st.caption("Returns shown as decimal fractions (e.g. 0.47 = 47% 1-year return).")
    else:
        st.info("No return columns found for performance chart.")


    # Similar stocks section
    st.markdown("#### 🧭 Similar stocks to this one")

    similar_df = find_similar_stocks(df, stock_row, top_n=5)
    if not similar_df.empty:
        st.dataframe(similar_df)
        st.caption(
            "These are numerically similar based on returns, volatility and score "
            "(educational only, not advice)."
        )
    else:
        st.info("Not enough numeric columns to compute similarity.")


# ========== RIGHT: stock card + chat + global search ==========
with right_col:
    st.markdown("### 2️⃣ Inspect stock card")

    with st.expander("Show full stock card (raw text)", expanded=False):
        st.text(stock_row["stock_card"])

    st.markdown("---")
    st.markdown("### 3️⃣ Chat about this stock")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a question about this stock…")

    if user_question:
        st.session_state["messages"].append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        if not client:
            with st.chat_message("assistant"):
                st.warning(
                    "Please enter your OpenAI API key in the left sidebar."
                )
        else:
            system_message = (
                "You are a cautious PSX stock advisor chatbot. "
                "You ONLY use the information given in the stock card text. "
                "You do NOT invent new financial facts or give buy/sell advice. "
                "You may interpret the ML probability and signal, but only as "
                "educational commentary. Always include a short disclaimer that "
                "this is not investment advice."
            )

            user_prompt = f"""
Here is the stock card for PSX ticker {selected_ticker}:

{stock_row["stock_card"]}

The user asks: {user_question}

Using ONLY the information in the stock card, answer the question in clear, simple language.
If the question cannot be answered from the card, say so explicitly.
"""

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_prompt},
                            ],
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as e:
                        st.error(f"Error while calling OpenAI API: {e}")

    st.markdown("---")
    st.markdown("### 4️⃣ Explore & search multiple stocks")

    search_text = st.text_input(
        "Search by ticker or keyword (e.g., 'low liquidity', 'volatility'):",
        value="",
    )

    if search_text.strip():
        search_lower = search_text.lower()
        mask = (
            df["ticker"].astype(str).str.lower().str.contains(search_lower)
            | df["stock_card"]
            .astype(str)
            .str.lower()
            .str.contains(search_lower)
        )
        results = df[mask].copy()

        cols_to_show = [
            c
            for c in [
                "ticker",
                "close",
                "return_1y",
                "return_3m",
                "return_1m",
                "vol_annual",
                "liquidity",
                "score",
            ]
            if c in results.columns
        ]

        if not results.empty and cols_to_show:
            st.write(f"Found **{len(results)}** matching stocks:")
            st.dataframe(results[cols_to_show])
        elif results.empty:
            st.info("No stocks matched your search text.")
        else:
            st.info("Matching stocks found but no numeric columns to show.")
    else:
        st.caption(search_tip)
