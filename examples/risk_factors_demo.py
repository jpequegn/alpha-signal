"""Demo script for Risk Factor Calculator.

This demonstrates how to use the RiskFactorCalculator to compute
4 independent risk factors for bubble detection.
"""

from src.daemon.risk_factors import RiskFactorCalculator


def main():
    """Demonstrate risk factor calculations."""
    calc = RiskFactorCalculator()

    # Historical P/E percentiles (example: S&P 500)
    pe_percentiles = {
        'p25': 18.0,  # 25th percentile
        'p75': 28.0,  # 75th percentile
        'p90': 35.0   # 90th percentile
    }

    print("=" * 60)
    print("RISK FACTOR CALCULATOR DEMO")
    print("=" * 60)
    print()

    # Scenario 1: Normal Market Conditions
    print("SCENARIO 1: Normal Market Conditions")
    print("-" * 60)
    print("Market Data:")
    print("  - P/E Ratio: 22.0")
    print("  - VIX: 18.0")
    print("  - Market Breadth: 75% (stocks above 200-day MA)")
    print("  - Weekly Price Change: +1.5%")
    print()

    val_risk = calc.calculate_valuation_risk(22.0, pe_percentiles)
    vol_risk = calc.calculate_volatility_risk(18.0)
    breadth_risk = calc.calculate_breadth_risk(75.0)
    mom_risk = calc.calculate_momentum_risk(1.5)

    result = calc.aggregate_risks(val_risk, vol_risk, breadth_risk, mom_risk)

    print("Risk Scores (0.0-0.8 scale):")
    print(f"  - Valuation Risk:  {val_risk:.3f}")
    print(f"  - Volatility Risk: {vol_risk:.3f}")
    print(f"  - Breadth Risk:    {breadth_risk:.3f}")
    print(f"  - Momentum Risk:   {mom_risk:.3f}")
    print(f"  - Average Risk:    {result['average_risk']:.3f}")
    print(f"  - Max Risk:        {result['max_risk']:.3f}")
    print()

    # Scenario 2: Bubble Warning Signs
    print("SCENARIO 2: Bubble Warning Signs")
    print("-" * 60)
    print("Market Data:")
    print("  - P/E Ratio: 42.0 (elevated)")
    print("  - VIX: 11.0 (complacency)")
    print("  - Market Breadth: 92% (broad participation)")
    print("  - Weekly Price Change: +11% (parabolic)")
    print()

    val_risk = calc.calculate_valuation_risk(42.0, pe_percentiles)
    vol_risk = calc.calculate_volatility_risk(11.0)
    breadth_risk = calc.calculate_breadth_risk(92.0)
    mom_risk = calc.calculate_momentum_risk(11.0)

    result = calc.aggregate_risks(val_risk, vol_risk, breadth_risk, mom_risk)

    print("Risk Scores (0.0-0.8 scale):")
    print(f"  - Valuation Risk:  {val_risk:.3f} ⚠️  (ELEVATED)")
    print(f"  - Volatility Risk: {vol_risk:.3f}")
    print(f"  - Breadth Risk:    {breadth_risk:.3f}")
    print(f"  - Momentum Risk:   {mom_risk:.3f} ⚠️  (PARABOLIC)")
    print(f"  - Average Risk:    {result['average_risk']:.3f}")
    print(f"  - Max Risk:        {result['max_risk']:.3f}")
    print()
    print("Analysis: High valuation + parabolic momentum suggest bubble risk")
    print()

    # Scenario 3: Market Crash
    print("SCENARIO 3: Market Crash")
    print("-" * 60)
    print("Market Data:")
    print("  - P/E Ratio: 16.0 (compressed)")
    print("  - VIX: 48.0 (extreme fear)")
    print("  - Market Breadth: 15% (severe decline)")
    print("  - Weekly Price Change: -13% (panic selling)")
    print()

    val_risk = calc.calculate_valuation_risk(16.0, pe_percentiles)
    vol_risk = calc.calculate_volatility_risk(48.0)
    breadth_risk = calc.calculate_breadth_risk(15.0)
    mom_risk = calc.calculate_momentum_risk(-13.0)

    result = calc.aggregate_risks(val_risk, vol_risk, breadth_risk, mom_risk)

    print("Risk Scores (0.0-0.8 scale):")
    print(f"  - Valuation Risk:  {val_risk:.3f}")
    print(f"  - Volatility Risk: {vol_risk:.3f} 🚨 (EXTREME)")
    print(f"  - Breadth Risk:    {breadth_risk:.3f} 🚨 (SEVERE)")
    print(f"  - Momentum Risk:   {mom_risk:.3f} 🚨 (PANIC)")
    print(f"  - Average Risk:    {result['average_risk']:.3f}")
    print(f"  - Max Risk:        {result['max_risk']:.3f}")
    print()
    print("Analysis: Multiple high-risk factors indicate market stress")
    print()

    print("=" * 60)
    print("Next Steps:")
    print("  1. These 4 risk scores will be passed to an LLM (Task 3.5)")
    print("  2. LLM synthesizes into bubble probability (0.0-1.0)")
    print("  3. Confidence adjustments applied (Task 3.6)")
    print("=" * 60)


if __name__ == "__main__":
    main()
