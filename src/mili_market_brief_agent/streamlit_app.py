import io
import os
import sys
import urllib.parse
import streamlit as st

OPENAI_KEY_PRESENT = bool(os.getenv("OPENAI_API_KEY"))

# Ensure the project root is on sys.path when streamlit runs this file as a script.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from mili_market_brief_agent.agent import run_personalized_market_brief_agent
except ImportError:
    from agent import run_personalized_market_brief_agent


# Synthetic client profiles for testing
SAMPLE_CLIENTS = {
    "Margaret Chen (Conservative, Retired)": {
        "risk_profile": "Conservative",
        "holdings": "BND, 150, 18750, Bonds\nDVY, 75, 7650, Dividend Stocks\nJNJ, 30, 5400, Healthcare\nPEP, 25, 3750, Consumer Staples\nT, 100, 2300, Utilities",
    },
    "James Rodriguez (Moderate, Mid-Career)": {
        "risk_profile": "Moderate",
        "holdings": "VOO, 60, 32400, Broad Market\nQQQ, 40, 15200, Tech\nVEA, 50, 3500, International\nVNQ, 25, 3250, Real Estate\nBND, 30, 3750, Bonds",
    },
    "Sarah Thompson (Growth, Young Professional)": {
        "risk_profile": "Growth",
        "holdings": "AAPL, 100, 18000, Technology\nMSFT, 50, 13000, Technology\nGOOGL, 30, 3900, Technology\nNVDA, 25, 11250, Semiconductors\nAMZN, 15, 3000, Consumer Discretionary",
    },
    "Robert Williams (Aggressive, Entrepreneur)": {
        "risk_profile": "Growth",
        "holdings": "NVDA, 75, 33750, Technology\nTSLA, 40, 12000, Automotive\nCRWD, 50, 8000, Cybersecurity\nAI, 30, 2700, Artificial Intelligence\nARKK, 100, 5300, Innovation ETF",
    },
}


def main() -> None:
    st.set_page_config(page_title="Mili Market Brief Agent", layout="wide")

    st.markdown(
        """
        <style>
        .css-1f4mp12, .css-1avcm0n, label {
            color: #7C62C4 !important;
        }
        .main > div > .block-container h2,
        .main > div > .block-container h3 {
            color: #7C62C4 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = os.path.join(os.path.dirname(__file__), "pics", "Mili Logo.svg")
    with open(logo_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    svg_data = urllib.parse.quote(svg_content)

    with st.container():
        st.markdown(
            f"""
            <div style='position:relative; margin-bottom:1rem; padding-top:0.5rem;'>
                <div style='position:absolute; top:0; left:0; z-index:10;'>
                    <img src='data:image/svg+xml;utf8,{svg_data}' style='width:75%; height:auto; max-height:none; display:block;' alt='Mili Logo' />
                </div>
            </div>
            <div style='display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:2rem; margin-bottom:1rem;'>
                <h2 style='color:#000000; text-align:center; margin:0.5rem 0 0.25rem;'>Mili Market Brief Agent</h2>
                <p style='text-align:center; margin:0;'>Use this demo agent to ingest a client's holdings, pull market data, and generate a personalized morning brief.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

    with st.sidebar.form(key="agent_input"):
        st.markdown("### Sample Clients")
        selected_client = st.selectbox(
            "Select a sample client profile",
            list(SAMPLE_CLIENTS.keys()),
            help="Choose a realistic client scenario to test with",
        )
        
        sample_data = SAMPLE_CLIENTS[selected_client]
        client_name = selected_client.split(" (")[0]
        risk_profile = st.selectbox("Client risk profile", ["Conservative", "Moderate", "Growth"], index=["Conservative", "Moderate", "Growth"].index(sample_data["risk_profile"]))
        
        if not OPENAI_KEY_PRESENT:
            st.warning("Set OPENAI_API_KEY in your environment to enable OpenAI Responses tool calls in the agent.")
        
        st.markdown("### Or Upload Custom Data")
        uploaded_file = st.file_uploader("Upload holdings CSV", type=["csv"])
        
        holdings_text = st.text_area(
            "Or paste holdings (ticker, quantity, market_value, sector)",
            sample_data["holdings"],
            height=200,
        )
        
        schedule = st.checkbox("Schedule for morning delivery", value=False)
        submit = st.form_submit_button("Generate Brief")

    if submit:
        csv_buffer = None
        if uploaded_file is not None:
            try:
                csv_bytes = uploaded_file.getvalue().decode("utf-8")
                csv_buffer = io.StringIO(csv_bytes)
            except Exception:
                st.error("Unable to read uploaded CSV file. Please check the format.")

        output = run_personalized_market_brief_agent(
            client_name=client_name,
            holdings_text=holdings_text,
            uploaded_file=csv_buffer,
            risk_profile=risk_profile,
            schedule=schedule,
        )

        st.header("Advisor-ready Summary")
        st.markdown("```")
        st.text(output["advisor_summary"])
        st.markdown("```")

        with st.expander("Show structured agent output and tool reasoning"):
            st.json(output)

        with st.expander("Agent workflow steps"):
            for step in output["agent_steps"]:
                st.write(step)


if __name__ == "__main__":
    main()
