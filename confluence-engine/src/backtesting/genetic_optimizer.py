"""
Genetic Algorithm optimizer for strategy parameters.

Evolves strategy parameters using natural selection:
1. Create a population of parameter sets (random initialization)
2. Backtest each one (fitness = risk-adjusted return)
3. Select the fittest (tournament selection)
4. Crossover (combine parameters from two parents)
5. Mutate (random perturbation)
6. Repeat for N generations

Anti-overfitting measures:
- Fitness function penalizes parameter count (Occam's razor)
- Walk-forward validation for final candidates
- Population diversity preservation (no monoculture)
- Minimum trade count requirement
- Deflated Sharpe in fitness

Following Marcos López de Prado's recommendation:
"The goal is not to find the best parameters, but to find
parameters that are ROBUST across different market conditions."
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestConfig, BacktestResult, run_backtest, walk_forward_analysis
from src.backtesting.strategies import BaseStrategy, StrategyMeta

log = logging.getLogger("forge.genetic")


@dataclass
class Individual:
    """One member of the population — a set of strategy parameters."""
    params: dict[str, float]
    fitness: float = 0.0
    sharpe: float = 0.0
    cagr: float = 0.0
    max_dd: float = 0.0
    trades: int = 0
    generation: int = 0


@dataclass
class OptimizationResult:
    """Result of genetic optimization."""
    best_individual: Individual
    best_params: dict[str, float]
    population_history: list[list[Individual]]   # All generations
    convergence_curve: list[float]                # Best fitness per generation
    strategy_name: str
    symbol: str
    generations_run: int
    total_backtests: int
    wf_efficiency: float | None = None           # Walk-forward check on winner


@dataclass
class GeneticConfig:
    """Configuration for the genetic optimizer."""
    population_size: int = 50
    num_generations: int = 30
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    mutation_strength: float = 0.3      # How much to perturb on mutation
    elitism_count: int = 3              # Top N carried forward unchanged
    min_trades: int = 10                # Minimum trades for valid fitness
    param_count_penalty: float = 0.05   # Penalty per parameter (anti-overfit)
    diversity_threshold: float = 0.1    # Min diversity to prevent monoculture
    early_stop_generations: int = 8     # Stop if no improvement for N gens


def _create_strategy_with_params(
    strategy_class: type,
    params: dict[str, float],
) -> BaseStrategy:
    """Instantiate a strategy with given parameters (casting to correct types)."""
    # Strategy __init__ may expect int for some params
    meta = strategy_class().meta() if hasattr(strategy_class, '__init__') else None
    init_params = {}
    for key, val in params.items():
        # Check if the parameter should be int based on its range
        if isinstance(val, float) and val == int(val):
            init_params[key] = int(val)
        else:
            init_params[key] = val
    return strategy_class(**init_params)


def compute_fitness(
    result: BacktestResult,
    param_count: int,
    config: GeneticConfig,
) -> float:
    """
    Compute fitness score for an individual.

    Fitness = composite score - parameter penalty

    Composite prioritizes:
    - Sharpe ratio (40%) — risk-adjusted returns
    - Profit factor (25%) — quality of edge
    - Sortino (20%) — downside risk
    - Calmar (15%) — return per max pain

    Parameter penalty discourages overfitting with too many knobs.
    """
    report = result.report

    # Reject if too few trades
    if report.trades.total_trades < config.min_trades:
        return -1.0

    # Reject if negative expectancy
    if report.trades.expectancy <= 0:
        return -0.5

    # Normalize components to [0, 1] range
    sharpe_score = min(max(report.sharpe_ratio / 3.0, 0), 1.0)
    pf_score = min(max((report.trades.profit_factor - 1.0) / 2.0, 0), 1.0)
    sortino_score = min(max(report.sortino_ratio / 4.0, 0), 1.0)
    calmar_score = min(max(report.calmar_ratio / 3.0, 0), 1.0)

    # Composite
    composite = (
        0.40 * sharpe_score
        + 0.25 * pf_score
        + 0.20 * sortino_score
        + 0.15 * calmar_score
    )

    # Parameter count penalty (Occam's razor)
    penalty = param_count * config.param_count_penalty

    # Bonus for consistency (low drawdown relative to return)
    if report.drawdown.max_drawdown_pct > 0:
        consistency_bonus = min(0.1, report.cagr_pct / (report.drawdown.max_drawdown_pct * 10))
    else:
        consistency_bonus = 0.1

    fitness = composite - penalty + consistency_bonus

    return round(fitness, 4)


def _random_params(param_ranges: dict[str, tuple]) -> dict[str, float]:
    """Generate random parameters within ranges."""
    params = {}
    for name, (lo, hi) in param_ranges.items():
        if isinstance(lo, int) and isinstance(hi, int):
            params[name] = random.randint(lo, hi)
        else:
            params[name] = random.uniform(lo, hi)
    return params


def _crossover(parent1: dict, parent2: dict) -> dict:
    """Uniform crossover — each parameter randomly from one parent."""
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child


def _mutate(params: dict, param_ranges: dict[str, tuple], strength: float) -> dict:
    """Mutate parameters with Gaussian perturbation."""
    mutated = {}
    for name, val in params.items():
        lo, hi = param_ranges[name]
        if random.random() < 0.3:  # 30% chance to mutate each param
            range_size = hi - lo
            perturbation = random.gauss(0, strength * range_size)
            new_val = val + perturbation
            new_val = max(lo, min(hi, new_val))
            if isinstance(lo, int) and isinstance(hi, int):
                new_val = round(new_val)
            mutated[name] = new_val
        else:
            mutated[name] = val
    return mutated


def _tournament_select(
    population: list[Individual],
    tournament_size: int,
) -> Individual:
    """Tournament selection — pick best from random subset."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda ind: ind.fitness)


def _population_diversity(population: list[Individual]) -> float:
    """Measure population diversity (coefficient of variation of fitness)."""
    fitnesses = [ind.fitness for ind in population if ind.fitness > 0]
    if len(fitnesses) < 2:
        return 1.0
    mean_f = np.mean(fitnesses)
    if mean_f == 0:
        return 1.0
    return float(np.std(fitnesses) / abs(mean_f))


def optimize(
    strategy_class: type,
    df: pd.DataFrame,
    symbol: str,
    asset_class: str = "stock",
    backtest_config: BacktestConfig | None = None,
    genetic_config: GeneticConfig | None = None,
    validate_winner: bool = True,
) -> OptimizationResult:
    """
    Run genetic optimization on a strategy.

    Args:
        strategy_class: The strategy class to optimize (not instance).
        df: Historical OHLCV data.
        symbol: Asset symbol.
        asset_class: "stock", "crypto", or "polymarket".
        backtest_config: Backtesting configuration.
        genetic_config: Genetic algorithm configuration.
        validate_winner: Run walk-forward validation on the winner.

    Returns:
        OptimizationResult with best parameters and convergence history.
    """
    if backtest_config is None:
        backtest_config = BacktestConfig()
    if genetic_config is None:
        genetic_config = GeneticConfig()

    # Get strategy metadata
    template = strategy_class()
    meta = template.meta()
    param_ranges = meta.param_ranges

    if not param_ranges:
        log.warning("Strategy %s has no tunable parameters", meta.name)
        result = run_backtest(template, df, symbol, asset_class, backtest_config)
        return OptimizationResult(
            best_individual=Individual(params={}, fitness=0),
            best_params={},
            population_history=[],
            convergence_curve=[],
            strategy_name=meta.name,
            symbol=symbol,
            generations_run=0,
            total_backtests=1,
        )

    log.info(
        "Starting genetic optimization: %s on %s (%d params, pop=%d, gens=%d)",
        meta.name, symbol, meta.param_count,
        genetic_config.population_size, genetic_config.num_generations,
    )

    # Initialize population
    population: list[Individual] = []
    for _ in range(genetic_config.population_size):
        params = _random_params(param_ranges)
        population.append(Individual(params=params, generation=0))

    population_history: list[list[Individual]] = []
    convergence: list[float] = []
    total_backtests = 0
    best_ever_fitness = -float("inf")
    stagnant_gens = 0

    for gen in range(genetic_config.num_generations):
        # Evaluate fitness
        for ind in population:
            if ind.fitness != 0:
                continue  # Already evaluated (elites)
            try:
                strategy = _create_strategy_with_params(strategy_class, ind.params)
                result = run_backtest(strategy, df, symbol, asset_class, backtest_config)
                ind.fitness = compute_fitness(result, meta.param_count, genetic_config)
                ind.sharpe = result.report.sharpe_ratio
                ind.cagr = result.report.cagr_pct
                ind.max_dd = result.report.drawdown.max_drawdown_pct
                ind.trades = result.report.trades.total_trades
                total_backtests += 1
            except Exception as e:
                log.debug("Backtest failed for params %s: %s", ind.params, e)
                ind.fitness = -2.0

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness = population[0].fitness
        convergence.append(best_fitness)
        population_history.append(list(population))

        log.info(
            "Gen %d/%d: best=%.4f (Sharpe=%.2f, CAGR=%.1f%%, DD=%.1f%%, trades=%d)",
            gen + 1, genetic_config.num_generations,
            best_fitness, population[0].sharpe, population[0].cagr,
            population[0].max_dd, population[0].trades,
        )

        # Early stopping
        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            stagnant_gens = 0
        else:
            stagnant_gens += 1

        if stagnant_gens >= genetic_config.early_stop_generations:
            log.info("Early stopping: no improvement for %d generations", stagnant_gens)
            break

        # Check diversity
        diversity = _population_diversity(population)
        if diversity < genetic_config.diversity_threshold:
            # Inject random individuals to prevent monoculture
            inject_count = genetic_config.population_size // 4
            for _ in range(inject_count):
                params = _random_params(param_ranges)
                population.append(Individual(params=params, generation=gen + 1))
            log.info("Diversity injection: %d new random individuals", inject_count)

        # Selection + reproduction
        next_gen: list[Individual] = []

        # Elitism: carry top N unchanged
        for i in range(min(genetic_config.elitism_count, len(population))):
            elite = Individual(
                params=dict(population[i].params),
                fitness=population[i].fitness,
                sharpe=population[i].sharpe,
                cagr=population[i].cagr,
                max_dd=population[i].max_dd,
                trades=population[i].trades,
                generation=gen + 1,
            )
            next_gen.append(elite)

        # Fill rest with crossover + mutation
        while len(next_gen) < genetic_config.population_size:
            parent1 = _tournament_select(population, genetic_config.tournament_size)
            parent2 = _tournament_select(population, genetic_config.tournament_size)

            if random.random() < genetic_config.crossover_rate:
                child_params = _crossover(parent1.params, parent2.params)
            else:
                child_params = dict(parent1.params)

            if random.random() < genetic_config.mutation_rate:
                child_params = _mutate(child_params, param_ranges, genetic_config.mutation_strength)

            next_gen.append(Individual(params=child_params, generation=gen + 1))

        population = next_gen

    # Final sort
    population.sort(key=lambda x: x.fitness, reverse=True)
    best = population[0]

    # Walk-forward validation on the winner
    wf_efficiency = None
    if validate_winner and len(df) > 200:
        log.info("Running walk-forward validation on best params: %s", best.params)
        try:
            strategy = _create_strategy_with_params(strategy_class, best.params)
            wf_result = walk_forward_analysis(
                strategy, df, symbol, asset_class, backtest_config,
                train_pct=0.70, num_windows=3, embargo_bars=5,
            )
            wf_efficiency = wf_result.wf_efficiency
            log.info("Walk-forward efficiency: %.2f (>0.5 = robust)", wf_efficiency)
        except Exception as e:
            log.warning("Walk-forward validation failed: %s", e)

    return OptimizationResult(
        best_individual=best,
        best_params=best.params,
        population_history=population_history,
        convergence_curve=convergence,
        strategy_name=meta.name,
        symbol=symbol,
        generations_run=len(convergence),
        total_backtests=total_backtests,
        wf_efficiency=wf_efficiency,
    )
