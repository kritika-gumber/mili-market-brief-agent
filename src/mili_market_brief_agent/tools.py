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
    """Construct an advisor-ready personalized market brief."""
    if not holdings:
        return "No client holdings were provided. Please upload or paste holdings data."

    primary_sectors = list({holding.sector for holding in holdings if holding.sector and holding.sector != "Unknown"})
    total_value = sum(holding.market_value for holding in holdings)
    top_assets = sorted(holdings, key=lambda h: h.market_value, reverse=True)[:3]

    lines = [
        f"Personalized Morning Brief for {client_name if client_name else 'your client'}",
        "---",
        f"Client profile summary: {risk_profile} risk appetite with ${total_value:,.0f} in flagged holdings.",
        "",
        "What moved today:",
    ]

    for mover in market_data["top_movers"]:
        lines.append(f"- {mover['ticker']}: {mover['move']} ({mover['reason']})")

    lines.append("")
    lines.append("Why it matters to the client:")
    if primary_sectors:
        lines.append(f"- The client's largest positions are concentrated in {', '.join(primary_sectors)}.")
    else:
        lines.append("- The client's holdings are diversified across multiple sectors.")

    if top_assets:
        lines.append(
            f"- Top exposures include {', '.join(asset.ticker for asset in top_assets)} representing the largest market value positions."
        )

    if market_data["headlines"]:
        lines.append("- Primary headlines to watch:")
        for headline in market_data["headlines"]:
            lines.append(f"  - {headline}")

    lines.append("")
    lines.append("Advisor talking points:")
    lines.append("- Confirm whether the client is comfortable with today's sector drivers and any recent performance volatility.")
    lines.append("- Discuss whether the allocation remains aligned with the client’s time horizon and risk preferences.")
    lines.append("- Consider whether any rebalancing or risk reduction is needed in high-concentration positions.")

    return "\n".join(lines)


def build_json_output(holdings: List[ClientHolding], market_data: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "holdings": [holding.__dict__ for holding in holdings],
        "market_data": market_data,
        "agent_steps": steps,
    }
