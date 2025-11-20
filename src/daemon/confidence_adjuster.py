"""Confidence adjuster that applies market risk to signal confidence.

This module adjusts signal confidence based on bubble probability,
reducing confidence during market extremes and bubbles.
"""

from typing import Any, Dict, Optional


class ConfidenceAdjuster:
    """Adjust signal confidence based on market bubble conditions."""

    def __init__(self):
        """Initialize confidence adjuster."""
        pass

    def adjust(
        self,
        original_confidence: float,
        bubble_probability: float
    ) -> float:
        """Adjust signal confidence by bubble probability.

        Formula: adjusted = original × (1 - bubble_probability)

        Examples:
            Normal market (bubble=0.2):   0.78 × 0.8 = 0.624
            Moderate bubble (bubble=0.5): 0.78 × 0.5 = 0.390
            Extreme bubble (bubble=0.8):  0.78 × 0.2 = 0.156

        Args:
            original_confidence: Original signal confidence (0.0-1.0)
            bubble_probability: Bubble probability from BubbleScorer (0.0-1.0)

        Returns:
            Adjusted confidence (0.0-1.0)

        Raises:
            ValueError: If inputs not in 0.0-1.0 range
        """
        self._validate_inputs(original_confidence, bubble_probability)

        # Apply adjustment formula
        adjusted = original_confidence * (1.0 - bubble_probability)

        # Ensure output is in valid range
        return max(0.0, min(1.0, adjusted))

    def adjust_signal(
        self,
        signal: Dict[str, Any],
        bubble_probability: float
    ) -> Dict[str, Any]:
        """Adjust signal and add risk context.

        Takes a Phase 2 signal and adjusts its confidence based on
        bubble conditions. Returns enriched signal with adjustment details.

        Args:
            signal: Signal dict with keys:
                - signal: 'BUY'|'SELL'|'HOLD'
                - confidence: original confidence (0.0-1.0)
                - reasoning: original reasoning text
                - [other fields preserved]
            bubble_probability: Bubble probability (0.0-1.0)

        Returns:
            Enriched signal dict with:
                - All original fields preserved
                - adjusted_confidence: new confidence
                - original_confidence: saved for reference
                - bubble_probability: market context
                - risk_adjusted_reasoning: reasoning with risk caveat

        Raises:
            ValueError: If signal missing required fields or inputs invalid
            KeyError: If signal missing 'confidence' field
        """
        # Validate signal structure
        if not isinstance(signal, dict):
            raise ValueError(f"Signal must be dict, got {type(signal)}")
        if "confidence" not in signal:
            raise KeyError("Signal must contain 'confidence' field")
        if "signal" not in signal:
            raise KeyError("Signal must contain 'signal' field")

        # Validate inputs
        self._validate_inputs(signal["confidence"], bubble_probability)

        # Calculate adjusted confidence
        adjusted_conf = self.adjust(signal["confidence"], bubble_probability)

        # Build enriched signal
        enriched_signal = signal.copy()
        enriched_signal["original_confidence"] = signal["confidence"]
        enriched_signal["adjusted_confidence"] = adjusted_conf
        enriched_signal["bubble_probability"] = bubble_probability
        enriched_signal["confidence"] = adjusted_conf  # Update main confidence

        # Add risk-adjusted reasoning
        enriched_signal["risk_adjusted_reasoning"] = self._build_risk_reasoning(
            signal.get("reasoning", ""),
            bubble_probability,
            adjusted_conf
        )

        return enriched_signal

    def get_confidence_adjustment_factor(self, bubble_probability: float) -> float:
        """Get the multiplier applied to confidence.

        Shows how much confidence is reduced.
        E.g., bubble_prob=0.5 means confidence multiplied by 0.5 (50% reduction).

        Args:
            bubble_probability: Bubble probability (0.0-1.0)

        Returns:
            Adjustment factor (0.0-1.0)
        """
        self._validate_inputs(1.0, bubble_probability)
        return 1.0 - bubble_probability

    def get_confidence_reduction(self, original_confidence: float, adjusted_confidence: float) -> float:
        """Get absolute confidence reduction.

        Shows how many percentage points confidence dropped.

        Args:
            original_confidence: Original confidence (0.0-1.0)
            adjusted_confidence: Adjusted confidence (0.0-1.0)

        Returns:
            Absolute reduction (e.g., 0.15 means 15% point reduction)
        """
        self._validate_inputs(original_confidence, 0.0)  # Check original only
        return max(0.0, original_confidence - adjusted_confidence)

    def assess_signal_reliability(
        self,
        adjusted_confidence: float
    ) -> str:
        """Assess signal reliability category based on adjusted confidence.

        Args:
            adjusted_confidence: Adjusted confidence (0.0-1.0)

        Returns:
            Reliability category: "Very High", "High", "Moderate", "Low", "Very Low"
        """
        if adjusted_confidence >= 0.8:
            return "Very High"
        elif adjusted_confidence >= 0.6:
            return "High"
        elif adjusted_confidence >= 0.4:
            return "Moderate"
        elif adjusted_confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def _validate_inputs(
        self,
        original_confidence: float,
        bubble_probability: float
    ) -> bool:
        """Validate that inputs are in 0.0-1.0 range.

        Args:
            original_confidence: Original confidence
            bubble_probability: Bubble probability

        Returns:
            True if valid

        Raises:
            ValueError: If inputs out of range
        """
        if not isinstance(original_confidence, (int, float)):
            raise ValueError(f"Confidence must be numeric, got {type(original_confidence)}")
        if not isinstance(bubble_probability, (int, float)):
            raise ValueError(f"Bubble probability must be numeric, got {type(bubble_probability)}")

        if not (0.0 <= original_confidence <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {original_confidence}")
        if not (0.0 <= bubble_probability <= 1.0):
            raise ValueError(f"Bubble probability must be 0.0-1.0, got {bubble_probability}")

        return True

    def _build_risk_reasoning(
        self,
        original_reasoning: str,
        bubble_probability: float,
        adjusted_confidence: float
    ) -> str:
        """Build risk-adjusted reasoning with context about market conditions.

        Args:
            original_reasoning: Original signal reasoning
            bubble_probability: Bubble probability (0.0-1.0)
            adjusted_confidence: Adjusted confidence after risk adjustment

        Returns:
            Risk-adjusted reasoning string
        """
        if not original_reasoning:
            original_reasoning = "No original reasoning provided"

        # Determine market regime description
        if bubble_probability < 0.2:
            regime = "healthy market conditions"
        elif bubble_probability < 0.4:
            regime = "slightly elevated risk conditions"
        elif bubble_probability < 0.6:
            regime = "moderate bubble risk"
        elif bubble_probability < 0.8:
            regime = "elevated bubble risk"
        else:
            regime = "extreme market conditions"

        # Add caveat if confidence significantly reduced
        reduction = 1.0 - (adjusted_confidence / max(0.001, 1.0 - bubble_probability))
        if reduction > 0.3:
            caveat = f" WARNING: Confidence significantly reduced due to {regime}."
        else:
            caveat = f" Signal adjusted for {regime}."

        return f"{original_reasoning}{caveat}"
