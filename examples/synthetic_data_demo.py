"""Demonstration of synthetic risk data generator.

This script showcases the SyntheticRiskDataGenerator generating realistic
market data for testing the bubble detection system.
"""

import numpy as np
from src.backfill.synthetic_data import SyntheticRiskDataGenerator


def print_statistics(name: str, data: np.ndarray):
    """Print summary statistics for a data series."""
    print(f"\n{name} Statistics:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Std Dev: {np.std(data):.2f}")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Max: {np.max(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")


def main():
    """Demonstrate synthetic data generation."""
    print("=" * 70)
    print("SYNTHETIC RISK DATA GENERATOR DEMONSTRATION")
    print("=" * 70)

    # Create generator with fixed seed for reproducibility
    gen = SyntheticRiskDataGenerator(seed=42, start_year=2015, end_year=2024)
    print(f"\nGenerator initialized:")
    print(f"  Period: {gen.start_year}-{gen.end_year} ({gen.num_years} years)")
    print(f"  Trading days: {gen.num_days}")

    # Generate all 4 risk factors
    print("\n" + "=" * 70)
    print("GENERATING ALL RISK FACTORS")
    print("=" * 70)

    price = gen._create_price_series(gen.num_days)
    pe_ratio = gen.generate_pe_ratio()
    vix = gen.generate_vix()
    breadth = gen.generate_breadth(price_series=price)
    momentum = gen.generate_momentum(price)

    # Print statistics
    print_statistics("Price (SPY-like)", price)
    print_statistics("P/E Ratio", pe_ratio)
    print_statistics("VIX", vix)
    print_statistics("Market Breadth (%)", breadth)
    print_statistics("Momentum (weekly %)", momentum)

    # Test scenarios
    print("\n" + "=" * 70)
    print("PRE-BUILT MARKET SCENARIOS")
    print("=" * 70)

    scenarios = ["normal", "bubble", "crash", "recovery"]

    for scenario in scenarios:
        print(f"\n{scenario.upper()} SCENARIO:")
        print("-" * 40)

        data = gen.generate_market_scenario(scenario, num_days=500)

        print(f"  P/E Ratio: {np.mean(data['pe_ratio']):.1f} (range: {np.min(data['pe_ratio']):.1f}-{np.max(data['pe_ratio']):.1f})")
        print(f"  VIX: {np.mean(data['vix']):.1f} (range: {np.min(data['vix']):.1f}-{np.max(data['vix']):.1f})")
        print(f"  Breadth: {np.mean(data['breadth']):.1f}% (range: {np.min(data['breadth']):.1f}-{np.max(data['breadth']):.1f}%)")
        print(f"  Momentum: {np.mean(data['momentum']):.1f}% (range: {np.min(data['momentum']):.1f}-{np.max(data['momentum']):.1f}%)")

    # Demonstrate reproducibility
    print("\n" + "=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)

    gen1 = SyntheticRiskDataGenerator(seed=123)
    gen2 = SyntheticRiskDataGenerator(seed=123)

    pe1 = gen1.generate_pe_ratio(100)
    pe2 = gen2.generate_pe_ratio(100)

    identical = np.array_equal(pe1, pe2)
    print(f"\nSame seed (123) produces identical results: {identical}")
    print(f"  First 5 values (gen1): {pe1[:5]}")
    print(f"  First 5 values (gen2): {pe2[:5]}")

    # Different seeds
    gen3 = SyntheticRiskDataGenerator(seed=456)
    pe3 = gen3.generate_pe_ratio(100)

    different = not np.array_equal(pe1, pe3)
    print(f"\nDifferent seeds (123 vs 456) produce different results: {different}")
    print(f"  First 5 values (seed=123): {pe1[:5]}")
    print(f"  First 5 values (seed=456): {pe3[:5]}")

    # Demonstrate temporal properties
    print("\n" + "=" * 70)
    print("TEMPORAL PROPERTIES")
    print("=" * 70)

    gen = SyntheticRiskDataGenerator(seed=42)
    pe = gen.generate_pe_ratio(5000)

    # Mean reversion
    print(f"\nP/E Ratio Mean Reversion:")
    print(f"  Long-term mean: {np.mean(pe):.2f} (target: 22.0)")
    print(f"  Autocorrelation (lag-1): {np.corrcoef(pe[:-1], pe[1:])[0, 1]:.3f}")

    # VIX jumps
    vix = gen.generate_vix(5000)
    high_vix_days = np.sum(vix > 35)
    print(f"\nVIX Jump Clustering:")
    print(f"  Days with VIX > 35: {high_vix_days} ({high_vix_days/5000*100:.1f}%)")
    print(f"  Max VIX spike: {np.max(vix):.1f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
