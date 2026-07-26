from suanet.reports import generalization_latex


def test_generalization_latex_contains_both_tables_and_pairs():
    text = generalization_latex(
        dataset_names=["BUSI", "BUSBRA"],
        cross_metrics={
            "BUSI->BUSBRA": {
                "accuracy": 0.8,
                "auc_roc": 0.9,
                "f1": 0.7,
                "sensitivity": 0.6,
                "specificity": 0.8,
                "precision": 0.75,
                "mcc": 0.5,
                "kappa": 0.4,
            }
        },
        within_auc={"BUSI": 0.95, "BUSBRA": 0.91},
    )
    assert text.count(r"\begin{table*}") == 2
    assert "BUSI & BUSBRA" in text
    assert r"\textbf{0.950}" in text
