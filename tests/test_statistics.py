from suanet.statistics import ablation_statistics, paired_cohens_d


def test_paired_effect_size_and_ablation_statistics():
    rows = {
        "Full Model": [
            {"auc_roc": value, "accuracy": value, "f1": value, "mcc": value}
            for value in [0.9, 0.91, 0.92, 0.93, 0.94]
        ],
        "w/o SVA": [
            {"auc_roc": value, "accuracy": value, "f1": value, "mcc": value}
            for value in [0.80, 0.82, 0.81, 0.85, 0.83]
        ],
        "w/o DLA": [
            {"auc_roc": value, "accuracy": value, "f1": value, "mcc": value}
            for value in [0.85, 0.87, 0.86, 0.90, 0.88]
        ],
    }
    result = ablation_statistics(rows)
    assert result["metrics"]["auc_roc"]["w/o SVA"]["mean_difference"] > 0
    assert result["metrics"]["auc_roc"]["w/o SVA"]["paired_t_p_bonferroni"] <= 1
    assert paired_cohens_d([1, 2, 3], [0, 1, 1]) > 0
