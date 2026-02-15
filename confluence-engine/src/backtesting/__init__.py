"""
Momentum Forge — Research-Grade Backtesting Engine.

Multi-asset day trading backtesting with:
- Combinatorial Purged Cross-Validation (CPCV) — gold standard anti-overfit
- Walk-forward optimization with embargo periods
- Triple barrier method for trade labeling
- Regime-filtered strategies (efficiency ratio + volatility)
- Kelly Criterion position sizing (half-Kelly default)
- Sharpe-weighted multi-strategy allocation
- Deflated Sharpe Ratio (corrected for multiple testing)
- Genetic parameter optimization
- Economic calendar guardrails
- AI-driven strategy evaluation (Claude + Codex)
- Stocks, Crypto, and Prediction Markets

Design principles (following Marcos López de Prado, Ernest Chan, Robert Pardo):
- CPCV over naive train/test — C(N,k) paths, not 1
- Regime detection before signal generation — skip choppy markets
- Penalize parameter count to prevent overfitting
- Out-of-sample verification before any strategy goes live
- Kelly sizing with half-Kelly conservative default
- Deflated Sharpe accounts for testing 300+ strategy combinations
"""
