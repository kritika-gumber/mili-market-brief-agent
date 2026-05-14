def generate_market_brief(topic: str) -> str:
    """Generate a placeholder market brief for the requested topic."""
    return (
        f"Market Brief for {topic}\n"
        "========================\n"
        "Summary: This is a starter market brief. Replace this text with real analysis and data.\n"
        "Key points: 1) market context, 2) competitive landscape, 3) risks and opportunities.\n"
    )


def main() -> None:
    topic = input("Enter a market topic: ")
    brief = generate_market_brief(topic.strip() or "General Market")
    print("\n" + brief)


if __name__ == "__main__":
    main()
