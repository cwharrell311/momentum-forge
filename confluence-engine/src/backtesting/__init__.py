"""
Momentum Forge — Research-Grade Backtesting Engine.

Multi-asset day trading backtesting with:
- Walk-forward optimization (anti-curve-fitting)
- Kelly Criterion position sizing
- Genetic parameter optimization
- Economic calendar guardrails
- AI-driven strategy evaluation (Claude + Codex)
- Stocks, Crypto, and Prediction Markets

Design principles (following Marcos López de Prado, Ernest Chan, Robert Pardo):
- Combinatorial purged cross-validation over naive train/test
- Walk-forward analysis with embargo periods
- Penalize parameter count to prevent overfitting
- Out-of-sample verification before any strategy goes live
- Kelly sizing with half-Kelly conservative default
"""
