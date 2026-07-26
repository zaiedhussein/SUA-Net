from suanet.data import make_folds, validate_no_group_leakage


def _samples():
    samples = []
    for group_index in range(10):
        label = group_index % 2
        for image_index in range(2):
            samples.append(
                {
                    "image": f"/tmp/{group_index}_{image_index}.png",
                    "label": label,
                    "sample_id": f"{group_index}_{image_index}",
                    "group_id": f"patient_{group_index}",
                }
            )
    return samples


def test_group_folds_have_no_leakage():
    folds = make_folds(_samples(), k=5, seed=42, strategy="stratified_group")
    assert len(folds) == 5
    for train, validation in folds:
        validate_no_group_leakage(train, validation)
        assert train
        assert validation


def test_image_folds_cover_every_sample_once_in_validation():
    samples = _samples()
    folds = make_folds(samples, k=5, seed=42, strategy="stratified_image")
    validation_ids = [sample["sample_id"] for _, validation in folds for sample in validation]
    assert sorted(validation_ids) == sorted(sample["sample_id"] for sample in samples)
