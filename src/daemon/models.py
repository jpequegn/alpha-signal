"""SQLAlchemy ORM models for daemon."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, ARRAY
from sqlalchemy.orm import relationship

from src.daemon.db import Base


class Signal(Base):
    """Signal decision with reasoning."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    signal = Column(String(10), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    confidence = Column(Float, nullable=False)
    key_factors = Column(ARRAY(String), nullable=True)
    contradictions = Column(ARRAY(String), nullable=True)
    final_reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    indicator_snapshot = relationship("IndicatorSnapshot", back_populates="signal", uselist=False)
    reasoning_steps = relationship("ReasoningStep", back_populates="signal")

    def __repr__(self):
        return f"<Signal({self.symbol}, {self.timestamp}, {self.signal}, {self.confidence})>"


class IndicatorSnapshot(Base):
    """Indicator values at time of signal."""

    __tablename__ = "indicator_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False)
    sma_20 = Column(Float, nullable=True)
    ema_12 = Column(Float, nullable=True)
    ema_26 = Column(Float, nullable=True)
    rsi_14 = Column(Float, nullable=True)
    macd_line = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_histogram = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    bb_bandwidth_pct = Column(Float, nullable=True)

    # Relationships
    signal = relationship("Signal", back_populates="indicator_snapshot")

    def __repr__(self):
        return f"<IndicatorSnapshot(signal_id={self.signal_id}, sma_20={self.sma_20})>"


class ReasoningStep(Base):
    """Per-indicator reasoning step."""

    __tablename__ = "reasoning_steps"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False)
    step_order = Column(Integer, nullable=False)  # 1=TREND, 2=MOMENTUM, 3=VOLATILITY
    indicator_group = Column(String(50), nullable=False)  # 'TREND', 'MOMENTUM', 'VOLATILITY'
    analysis = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    signal = relationship("Signal", back_populates="reasoning_steps")

    def __repr__(self):
        return f"<ReasoningStep({self.signal_id}, {self.indicator_group})>"
