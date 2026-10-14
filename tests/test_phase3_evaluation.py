"""
test_phase3_evaluation.py
-------------------------
Pytest unit and integration tests for Phase 3:
- Metric implementations (RMSE, MAE, Precision@K, Recall@K, MAP@K, NDCG@K, ILD)
- Leak-free train/test split verification
- RecommenderEvaluator pipeline execution and schema validation
"""

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.evaluator import (
    RecommenderEvaluator,
    compute_rmse,
    compute_mae,
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
    intra_list_diversity,
)


# =====================================================================
# METRIC UNIT TESTS
# =====================================================================

def test_rmse_and_mae():
    """Verify RMSE and MAE calculations on known vectors."""
    y_true = np.array([4.0, 3.0, 5.0, 2.0], dtype=np.float32)
    y_pred = np.array([4.0, 3.0, 4.0, 3.0], dtype=np.float32)

    # Errors: [0, 0, -1, 1], squared errors: [0, 0, 1, 1] -> mean 0.5 -> sqrt(0.5) ~ 0.7071
    # Absolute errors: [0, 0, 1, 1] -> mean 0.5
    assert np.isclose(compute_rmse(y_true, y_pred), np.sqrt(0.5), atol=1e-4)
    assert np.isclose(compute_mae(y_true, y_pred), 0.5, atol=1e-4)


def test_ranking_metrics_perfect_order():
    """Verify Precision, Recall, MAP, and NDCG for perfect recommendations."""
    recommended = [10, 20, 30, 40, 50]
    relevant = {10, 20, 30}

    assert precision_at_k(recommended, relevant, k=3) == 1.0
    assert recall_at_k(recommended, relevant, k=3) == 1.0
    assert average_precision_at_k(recommended, relevant, k=3) == 1.0
    assert ndcg_at_k(recommended, relevant, k=3) == 1.0


def test_ranking_metrics_zero_hits():
    """Verify ranking metrics when no recommended item is relevant."""
    recommended = [1, 2, 3]
    relevant = {4, 5, 6}

    assert precision_at_k(recommended, relevant, k=3) == 0.0
    assert recall_at_k(recommended, relevant, k=3) == 0.0
    assert average_precision_at_k(recommended, relevant, k=3) == 0.0
    assert ndcg_at_k(recommended, relevant, k=3) == 0.0


def test_ranking_metrics_partial_hits():
    """Verify ranking metrics on partial and lower-ranked hits."""
    recommended = [10, 99, 20]  # items 10 and 20 are relevant
    relevant = {10, 20, 30, 40}

    p = precision_at_k(recommended, relevant, k=3)
    r = recall_at_k(recommended, relevant, k=3)
    assert np.isclose(p, 2 / 3)
    assert np.isclose(r, 2 / 4)

    # Rank 1 hit (P=1/1), Rank 2 miss, Rank 3 hit (P=2/3)
    # AP@3 = (1.0 + 2/3) / min(3, 4) = (5/3) / 3 = 5/9
    ap = average_precision_at_k(recommended, relevant, k=3)
    assert np.isclose(ap, 5 / 9, atol=1e-4)


def test_intra_list_diversity():
    """Verify intra-list diversity computation."""
    # Create two orthogonal unit vectors -> cosine sim = 0, distance = 1.0
    embs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    tmdb_to_idx = {100: 0, 200: 1}

    div = intra_list_diversity([100, 200], embs, tmdb_to_idx)
    assert np.isclose(div, 1.0, atol=1e-4)

    # Two identical vectors -> distance = 0.0
    embs_same = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)
    div_same = intra_list_diversity([100, 200], embs_same, tmdb_to_idx)
    assert np.isclose(div_same, 0.0, atol=1e-4)


# =====================================================================
# EVALUATOR PIPELINE INTEGRATION TESTS
# =====================================================================

@pytest.fixture(scope="module")
def evaluator():
    """Shared RecommenderEvaluator on 80/20 reproducible split."""
    return RecommenderEvaluator(test_size=0.2, random_state=42)


def test_evaluator_split_leak_free(evaluator):
    """Verify that train and test splits are strictly disjoint."""
    train_idx = set(evaluator.train_df.index)
    test_idx = set(evaluator.test_df.index)

    # Zero overlap in split indices
    assert len(train_idx & test_idx) == 0

    # Test size matches expected fraction
    total_samples = len(evaluator.ratings_df)
    assert len(evaluator.train_df) + len(evaluator.test_df) == total_samples
    assert np.isclose(len(evaluator.test_df) / total_samples, 0.2, atol=0.01)


def test_evaluator_rating_prediction(evaluator):
    """Verify rating prediction evaluation table schema and baseline comparison."""
    df_rating = evaluator.evaluate_rating_prediction()

    assert isinstance(df_rating, pd.DataFrame)
    assert set(df_rating.columns) == {"Model", "RMSE", "MAE", "Test_Samples"}
    assert len(df_rating) >= 3

    # Extract RMSEs
    rmse_map = dict(zip(df_rating["Model"], df_rating["RMSE"]))
    assert "Global Mean Baseline" in rmse_map
    assert "SVD Collaborative Filtering" in rmse_map

    # SVD must achieve lower error than the naive global-mean baseline
    assert rmse_map["SVD Collaborative Filtering"] < rmse_map["Global Mean Baseline"]


def test_evaluator_ranking_benchmark(evaluator):
    """Verify ranking evaluation table schema, metric boundaries, and multi-model coverage."""
    # Fast evaluation on k=5
    df_ranking = evaluator.evaluate_ranking(k=5, min_test_positives=3)

    assert isinstance(df_ranking, pd.DataFrame)
    required_cols = {
        "Model",
        "Precision@5",
        "Recall@5",
        "MAP@5",
        "NDCG@5",
        "Candidate_Coverage_Pct",
        "Catalog_Coverage_Pct",
        "Intra_List_Diversity",
        "Evaluated_Users",
    }
    assert required_cols.issubset(df_ranking.columns)

    models_evaluated = set(df_ranking["Model"].tolist())
    assert "Popularity Baseline" in models_evaluated
    assert "TF-IDF Content Recommender" in models_evaluated
    assert "SVD Collaborative Filtering" in models_evaluated
    assert "Hybrid Recommender" in models_evaluated

    # Bounded metric validation across all models
    for _, row in df_ranking.iterrows():
        assert 0.0 <= row["Precision@5"] <= 1.0
        assert 0.0 <= row["Recall@5"] <= 1.0
        assert 0.0 <= row["MAP@5"] <= 1.0
        assert 0.0 <= row["NDCG@5"] <= 1.0
        assert 0.0 <= row["Candidate_Coverage_Pct"] <= 100.0
        assert 0.0 <= row["Catalog_Coverage_Pct"] <= 100.0
        assert 0.0 <= row["Intra_List_Diversity"] <= 2.0
        assert row["Evaluated_Users"] > 0


def test_evaluator_run_full_benchmark_saves_report(evaluator, tmp_path):
    """Verify that run_full_benchmark saves a valid markdown report file."""
    report_file = tmp_path / "test_report.md"
    result = evaluator.run_full_benchmark(k=5, save_report_path=report_file)

    assert "rating_prediction" in result
    assert "ranking" in result
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Moviq Recommender System - Empirical Evaluation Report" in content
    assert "Popularity Baseline" in content
    assert "Hybrid Recommender" in content
