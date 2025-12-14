import os
import streamlit as st
import pandas as pd
from openai import OpenAI

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="PSX AI Stock Learning Assistant",
    layout="wide",
)

CSV_FILE = "psx_stock_cards_ml.csv"

# Global OpenAI client (set after API key input)
client = None


# ============== HELPERS ==============
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_FILE)
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_FILE, encoding="latin1")
    return df


def clean_text(text):
    """Fix common mojibake issues from CSV (e.g. â etc.)."""
    if not isinstance(text, str):
        return text
    replacements = {
        # Remove / fix jumbled “Liquidity Warning” prefix
        "â ï¸ Liquidity Warning:": "Liquidity warning:",
        "â ï¸": "",
        "â": "–",
        "â": "—",
        "Â": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def make_display_name(row):
    ticker = str(row.get("ticker", "")).upper()
    name = str(row.get("company_name", ticker))
    return f"{ticker} | {name}"


def fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "N/A"


def fmt_float3(x):
    if pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "N/A"


def fmt_prob2(x):
    if pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"


def compute_risk_snapshot(row):
    """
    Heuristic risk snapshot based on annual volatility + liquidity.
    Returns (level, score in [0,1], reason_text).
    """
    vol = row.get("vol_annual", None)
    liq = str(row.get("liquidity", "")).lower()

    level = "Medium"
    score = 0.5

    if vol is not None and not pd.isna(vol):
        try:
            v = float(vol)
            if v < 0.15:
                level = "Lower"
                score = 0.25
            elif v < 0.25:
                level = "Medium"
                score = 0.5
            else:
                level = "Higher"
                score = 0.8
        except Exception:
            pass

    reason = "based on historical volatility"
    if "low" in liq:
        score = min(score + 0.1, 1.0)
        reason = "due to low liquidity"
    elif "high" in liq:
        score = max(score - 0.1, 0.0)
        reason = "with strong liquidity"

    return level, score, reason


# ============== STATE HELPERS ==============
def clear_conversation():
    st.session_state["chat_history"] = []


# ============== QUIZ (RUN ONCE) ==============
def show_profile_quiz():
    st.title("Investor Learning Profile")

    st.write(
        "Please answer a short set of questions so explanations can be aligned "
        "with your comfort level. This tool is strictly for educational use."
    )

    # Disclaimer on first page (where profile is set)
    st.markdown("#### ⚠️ Disclaimer")
    st.write(
        "This is an **AI-based educational tool** using historical PSX/KSE-100 data (2014- 2024). "
        "It does **not** provide investment advice, recommendations, or forecasts."
    )
    st.write(
        "- AI-generated text may be incomplete or inaccurate.\n"
        "- It uses historical patterns and does not predict the future.\n"
        "- For any investment decision, please consult a licensed advisor."
    )

    # Q1 – label is the full question (same style as selections)
    q2 = st.radio(
        "How would you feel if your investment temporarily declined by 20%?",
        [
            "Very uncomfortable – I prefer safety",
            "Somewhat concerned, but acceptable",
            "Comfortable if long-term return is higher",
        ],
    )

    # Q2 – label is the full question
    q3 = st.radio(
        "What is your main objective while using this tool?",
        [
            "Learn the basics and avoid large losses",
            "Balance stability and return potential",
            "Maximise return while accepting higher risk",
        ],
    )

    if st.button("Save profile and continue"):
        score = 0

        if q2.startswith("Very uncomfortable"):
            score += 0
        elif q2.startswith("Somewhat"):
            score += 1
        else:
            score += 2

        if q3.startswith("Learn"):
            score += 0
        elif q3.startswith("Balance"):
            score += 1
        else:
            score += 2

        if score <= 1:
            profile = "Conservative"
        elif score <= 3:
            profile = "Balanced"
        else:
            profile = "Aggressive"

        st.session_state["risk_profile"] = profile
        st.session_state["quiz_done"] = True

        st.success(f"Your learning profile has been set to: {profile}.")


# ============== AI EXPLANATIONS ==============
def generate_stock_summary(row, profile):
    """
    One-shot educational summary about the stock (used under Risk snapshot / explanation).
    """
    ticker = str(row.get("ticker", "")).upper()
    name = str(row.get("company_name", ticker))
    card_raw = str(row.get("stock_card", ""))
    card = clean_text(card_raw)

    if client is None:
        return (
            f"This is an educational summary for {name} ({ticker}).\n\n"
            "AI explanation is disabled because no valid API key is configured.\n"
            "Use the return trends, volatility measures and model tags as examples "
            "to understand how risk and behaviour can differ across stocks. "
            "This is for learning only."
        )

    prompt = f"""
Explain this PSX stock in clear, professional language for a beginner.
Do NOT give investment advice. Only educational explanation.

User profile: {profile}
Company: {name} ({ticker})

Stock summary text:
{card}

Explain in brief:
- What type of business this is (at a high level)
- How a beginner should interpret its returns and volatility
- How a {profile} learner might think about the risk–return balance
- How diversification (not putting everything in one stock) applies here
- End with: "This is for education only."
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You explain financial concepts simply and professionally for beginners. Never give advice.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        return (
            f"Educational overview for {name} ({ticker}). AI service unavailable.\n"
            "Use the displayed metrics as an example of risk–return trade-offs. "
            "This is for learning only."
        )


def answer_stock_question(row, profile, question):
    """
    Question-answer style chat about the selected stock.
    """
    ticker = str(row.get("ticker", "")).upper()
    name = str(row.get("company_name", ticker))
    card_raw = str(row.get("stock_card", ""))
    card = clean_text(card_raw)

    if client is None:
        return (
            "AI chat is disabled because no API key is configured. "
            "You can still review the metrics and stock card for learning."
        )

    prompt = f"""
You are an educational assistant for PSX investors.
Do NOT give investment advice. Only explain concepts and data.

User profile: {profile}
Company: {name} ({ticker})

Stock card:
{card}

User question about this stock:
{question}

Answer in a concise, structured way for a beginner. Do not tell the user to buy or sell.
End with: "This is for education only."
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Explain things clearly in simple, professional language without giving advice.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        return "Unable to generate a response at the moment. Please try again later."


# ============== LEFT PANE (API & CONTROLS) ==============
def render_left_pane():
    global client

    st.markdown("### 🔐 API & Controls")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxxxxxx",
    )

    if api_key:
        try:
            os.environ["OPENAI_API_KEY"] = api_key
            client = OpenAI()
            st.success("AI is enabled for this session.")
        except Exception:
            client = None
            st.error("Invalid API key. AI will remain disabled.")
    else:
        st.info("AI explanation and chat are disabled (no API key provided).")

    # Clear conversation
    if st.button("🧹 Clear conversation"):
        clear_conversation()
        st.success("Conversation cleared.")

    st.markdown("---")

    st.markdown("### 🧑‍🤝‍🧑 Need Human Assistance?")
    if st.button("📞 Proceed to human advisor"):
        st.info(
            "For actual investment decisions, please contact your relationship manager "
            "or a licensed financial advisor through formal bank channels."
        )

    st.markdown("---")

    # Investor Profile section
    st.markdown("### 👤 Investor Profile")
    profile = st.session_state.get("risk_profile", "Conservative")
    st.write(f"Your profile is set to **{profile}**.")

    if st.button("Change your profile"):
        st.session_state["quiz_done"] = False
        st.experimental_rerun()

    st.markdown("---")

    st.markdown("### Feedback")
    st.text_area("Share your comments (optional):")
    st.button("Submit feedback")


# ============== MAIN APP ==============
def main():
    # Quiz gate
    if "quiz_done" not in st.session_state:
        st.session_state["quiz_done"] = False
    # Chat history as list of {"role": "...", "content": "..."}
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["quiz_done"]:
        show_profile_quiz()
        return

    df = load_data()

    left_col, right_col = st.columns([1, 2])

    # LEFT PANE
    with left_col:
        render_left_pane()

    # RIGHT PANE
    with right_col:
        profile = st.session_state.get("risk_profile", "Conservative")

        # ----- Step 1: Choose a stock -----
        st.markdown("### 1️⃣ Choose a stock")

        search = st.text_input(
            "Select PSX stock:",
            placeholder="Type ticker or company name...",
        )

        filtered = df.copy()
        if search:
            q = search.lower()
            filtered = filtered[
                filtered["ticker"].astype(str).str.lower().str.contains(q)
                | filtered["company_name"].astype(str).str.lower().str.contains(q)
            ]

        if filtered.empty:
            st.warning("No matching companies found.")
            return

        options = filtered.index.tolist()
        labels = [make_display_name(filtered.loc[idx]) for idx in options]

        # Non-empty label + hide visually to avoid Streamlit warning
        selected_idx = st.selectbox(
            "Select stock",
            options,
            format_func=lambda x: labels[options.index(x)],
            label_visibility="collapsed",
        )

        row = filtered.loc[selected_idx]
        ticker = str(row.get("ticker", "")).upper()
        name = str(row.get("company_name", ticker))

        st.markdown(f"You selected: `{ticker}`")

        st.markdown("---")

        # ----- Step 2: Inspect stock card + Risk snapshot -----
        st.markdown("### 2️⃣ Inspect stock card")

        with st.expander("Show full stock card", expanded=False):
            st.write(clean_text(row.get("stock_card", "")))

        st.markdown("#### Risk snapshot")

        risk_level, risk_score, risk_reason = compute_risk_snapshot(row)

        st.write(f"Risk level: **{risk_level}** ({risk_reason})")
        st.progress(risk_score)

        # Numeric snapshot
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("1-month return", fmt_pct(row.get("return_1m")))
        with col2:
            st.metric("3-month return", fmt_pct(row.get("return_3m")))
        with col3:
            st.metric("1-year return", fmt_pct(row.get("return_1y")))

        col4, col5 = st.columns(2)
        with col4:
            st.metric("30-day volatility", fmt_float3(row.get("vol_30d")))
        with col5:
            st.metric("Annual volatility", fmt_float3(row.get("vol_annual")))

        st.write(
            f"Trend tag: `{row.get('trend', 'N/A')}`, "
            f"Liquidity: `{row.get('liquidity', 'N/A')}`"
        )

        # Educational summary now collapsible
        with st.expander("Educational summary", expanded=False):
            summary_text = generate_stock_summary(row, profile)
            st.write(summary_text)

        st.markdown("---")

        # ----- Step 3: Chat about this stock (ChatGPT-style) -----
         # ----- Step 3: Chat about the stock (ChatGPT-style) -----
        st.markdown("### 3️⃣ Chat about the stock")

        # Disclaimer directly under the heading
        st.info(
            "**Disclaimer:** This conversation is for **educational purposes only** "
            "and **not investment advice**. It is based on historical PSX data. "
            "For any investment decision, please contact a **licensed human advisor**."
        )

        # 1) Render existing messages (older first, newer last)
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 2) Chat input at the bottom (like ChatGPT)
        user_q = st.chat_input("Ask a question about this stock…")

        if user_q:
            # Append user message to history
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_q}
            )

            # Generate answer
            answer = answer_stock_question(row, profile, user_q)

            # Append assistant message to history
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
if __name__ == "__main__":
    main()
