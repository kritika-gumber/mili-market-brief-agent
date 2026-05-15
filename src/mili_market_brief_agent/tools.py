import csv
import io
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ClientHolding:
    ticker: str
    quantity: float
    market_value: float
    sector: str


def parse_holdings_text(raw_text: str) -> List[ClientHolding]:
    """Parse simple holdings text into a structured list."""
    holdings = []
    for line in raw_text.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 3:
            continue
        ticker = parts[0].upper()
        try:
            quantity = float(parts[1])
        except ValueError:
            quantity = 0.0
        try:
            market_value = float(parts[2])
        except ValueError:
            market_value = 0.0
        sector = parts[3] if len(parts) > 3 else "Unknown"
        holdings.append(ClientHolding(ticker=ticker, quantity=quantity, market_value=market_value, sector=sector))
    return holdings


def parse_holdings_csv(uploaded_csv: io.StringIO) -> List[ClientHolding]:
    reader = csv.DictReader(uploaded_csv)
    holdings = []
    for row in reader:
        ticker = row.get("ticker", row.get("symbol", "")).upper()
        if not ticker:
            continue
        try:
            quantity = float(row.get("quantity", row.get("shares", 0)))
        except (ValueError, TypeError):
            quantity = 0.0
        try:
            market_value = float(row.get("market_value", row.get("value", 0)))
        except (ValueError, TypeError):
            market_value = 0.0
        sector = row.get("sector", "Unknown")
        holdings.append(ClientHolding(ticker=ticker, quantity=quantity, market_value=market_value, sector=sector))
    return holdings


def fetch_market_data(sectors: List[str]) -> Dict[str, Any]:
    """Return mocked market data and news for the sectors referenced by the client."""
    default_data = {
        "top_movers": [
            {"ticker": "AAPL", "move": "+2.2%", "reason": "strong earnings reaction"},
            {"ticker": "TSLA", "move": "-1.4%", "reason": "autonomy guidance pressure"},
            {"ticker": "MSFT", "move": "+1.1%", "reason": "AI spending commentary"},
        ],
        "headlines": [
            "Equities rallied after U.S. inflation data came in cooler than expected.",
            "Large-cap tech names are driving most of the market gains today.",
        ],
    }

    sector_specific = []
    if "Technology" in sectors or "Tech" in sectors:
        sector_specific.append("Tech shares remain in focus after the earnings season.")
    if "Energy" in sectors:
        sector_specific.append("Energy names are reacting to oil price stabilization.")
    if "Financials" in sectors:
        sector_specific.append("Bank earnings are being watched for margin commentary.")

    return {
        "top_movers": default_data["top_movers"],
        "headlines": default_data["headlines"] + sector_specific,
        "sector_coverage": list({s for s in sectors if s != "Unknown"})[:3],
    }


def build_personalized_summary(
    client_name: str,
    holdings: List[ClientHolding],
    market_data: Dict[str, Any],
    risk_profile: str,
) -> str:
    """Construct an advisor-ready personalized market brief with client-specific insights."""
    if not holdings:
        return "No client holdings were provided. Please upload or paste holdings data to generate a brief."

    primary_sectors = list({h.sector for h in holdings if h.sector and h.sector != "Unknown"})
    total_value = sum(h.market_value for h in holdings)
    top_assets = sorted(holdings, key=lambda h: h.market_value, reverse=True)[:3]
    top_tickers = ", ".join(a.ticker for a in top_assets)
    sector_str = ", ".join(primary_sectors) if primary_sectors else "diversified sectors"
    risk_label = risk_profile.lower()

    lines = [
        f"Morning Brief — {client_name or 'Client'}",
        f"Risk profile: {risk_profile}  |  Portfolio value: ${total_value:,.0f}",
        "=" * 60,
        "",
        "MARKET SNAPSHOT",
    ]

    for mover in market_data["top_movers"]:
        lines.append(f"  {mover['ticker']:6s} {mover['move']:>7s}  — {mover['reason']}")

    lines += [
        "",
        "WHAT THIS MEANS FOR THE CLIENT",
        f"  {client_name or 'The client'} holds a {risk_label}-risk portfolio concentrated in {sector_str}.",
        f"  Largest positions are {top_tickers}, which together represent the bulk of market value exposure.",
    ]

    for headline in market_data.get("headlines", []):
        lines.append(f"  • {headline}")

    # Risk-profile-specific framing
    if risk_profile == "Conservative":
        risk_note = "Focus on capital preservation — flag any outsized moves in fixed income or dividend names."
    elif risk_profile == "Moderate":
        risk_note = "Monitor sector concentration; consider whether recent volatility warrants any tactical trim."
    else:  # Growth / Aggressive
        risk_note = "Growth-oriented positioning may amplify gains and losses — ensure client is aligned on drawdown tolerance."

    lines += [
        "",
        "ADVISOR TALKING POINTS",
        f"  1. {risk_note}",
        f"  2. Check whether {top_tickers} exposure is still within target weight given today's moves.",
        f"  3. Ask the client if any upcoming liquidity needs should inform near-term positioning.",
        "",
        "Prepared by Mili Market Brief Agent.",
    ]

    if not any(s for s in primary_sectors if s):
        lines.append("Note: Sector data was incomplete — review holdings for missing sector tags.")

    return "\n".join(lines)


def build_json_output(holdings: List[ClientHolding], market_data: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "holdings": [holding.__dict__ for holding in holdings],
        "market_data": market_data,
        "agent_steps": steps,
    }
