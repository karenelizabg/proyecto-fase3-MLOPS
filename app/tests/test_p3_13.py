"""P3-13 (#20): métricas, análisis de errores y candado de la evaluación final.

Las pruebas de métricas/análisis son puras (fixture calculado a mano); las de
`final.py` no entrenan ni tocan MLflow (fallan en el candado antes).
"""

import pytest

from evaluation.analysis import analyze
from evaluation.contracts import CLASSES, Prediction
from evaluation.metrics import evaluate
from final import MANIFEST_SPLITS, _source_images
from final import main as final_main
from recompute import sklearn_metrics


def a_prediction(
    true_label: int,
    predicted_label: int,
    crop_id: str = "c",
    source_image_id: str = "1",
) -> Prediction:
    probabilities = [0.9, 0.1] if predicted_label == 0 else [0.1, 0.9]
    return Prediction(
        crop_id=crop_id,
        source_image_id=source_image_id,
        true_label=true_label,
        predicted_label=predicted_label,
        probabilities=probabilities,
    )


def fixture_predictions() -> list[Prediction]:
    # Confusión real x predicha: [[3, 1], [2, 4]] (cat=0, dog=1).
    pairs = [(0, 0)] * 3 + [(0, 1)] + [(1, 1)] * 4 + [(1, 0)] * 2
    return [
        a_prediction(true, pred, crop_id=f"c{index:02d}")
        for index, (true, pred) in enumerate(pairs)
    ]


# --- métricas -------------------------------------------------------------------


def test_evaluate_hand_computed_metrics():
    metrics = evaluate(fixture_predictions(), CLASSES)
    assert metrics.confusion_matrix == [[3, 1], [2, 4]]
    assert metrics.total == 10
    assert metrics.accuracy == pytest.approx(0.7)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 8 / 11) / 2)

    cat = metrics.per_class["cat"]
    dog = metrics.per_class["dog"]
    assert cat.precision == pytest.approx(0.6)
    assert cat.recall == pytest.approx(0.75)
    assert cat.support == 4
    assert dog.precision == pytest.approx(0.8)
    assert dog.recall == pytest.approx(4 / 6)
    assert dog.support == 6


def test_changing_a_prediction_changes_the_metrics():
    predictions = fixture_predictions()
    before = evaluate(predictions, CLASSES)
    predictions[0] = a_prediction(0, 1, crop_id="c00")
    after = evaluate(predictions, CLASSES)
    assert after.accuracy != before.accuracy
    assert after.confusion_matrix != before.confusion_matrix


def test_confusion_matrix_sums_to_total():
    metrics = evaluate(fixture_predictions(), CLASSES)
    assert sum(sum(row) for row in metrics.confusion_matrix) == metrics.total


# --- análisis -------------------------------------------------------------------


def test_analysis_reports_baseline_most_confused_and_recall():
    predictions = fixture_predictions()
    metrics = evaluate(predictions, CLASSES)
    analysis = analyze(predictions, metrics)
    assert analysis.baseline_majority_accuracy == pytest.approx(0.6)
    assert analysis.most_confused_class == "dog"
    assert analysis.recall_per_class == pytest.approx({"cat": 0.75, "dog": 4 / 6})
    assert analysis.accuracy_hides_low_recall is False
    assert len(analysis.successes) == 6  # tope de 6
    assert len(analysis.errors) == 3


def test_analysis_flags_a_low_recall_hidden_by_accuracy():
    predictions = [a_prediction(0, 0, crop_id=f"c{i}") for i in range(18)]
    predictions.append(a_prediction(0, 1, crop_id="c18"))
    predictions.append(a_prediction(1, 0, crop_id="c19"))  # dog nunca acierta
    metrics = evaluate(predictions, CLASSES)
    analysis = analyze(predictions, metrics)
    assert metrics.accuracy == pytest.approx(0.9)
    assert metrics.per_class["dog"].recall == 0.0
    assert analysis.accuracy_hides_low_recall is True


# --- recompute (scikit-learn) ---------------------------------------------------


def test_recompute_matches_the_pure_metrics():
    predictions = fixture_predictions()
    metrics = evaluate(predictions, CLASSES)
    recomputed = sklearn_metrics(
        [p.true_label for p in predictions], [p.predicted_label for p in predictions]
    )
    assert recomputed["accuracy"] == pytest.approx(metrics.accuracy)
    assert recomputed["macro_f1"] == pytest.approx(metrics.macro_f1)
    assert recomputed["confusion_matrix"] == metrics.confusion_matrix
    assert recomputed["total"] == metrics.total
    for name in CLASSES:
        assert recomputed["per_class"][name]["recall"] == pytest.approx(
            metrics.per_class[name].recall
        )
        assert recomputed["per_class"][name]["support"] == metrics.per_class[name].support


# --- candado (sin tocar MLflow ni el modelo) ------------------------------------


def test_final_test_split_refuses_without_selection(tmp_path):
    rc = final_main(
        [
            "--split",
            "test",
            "--selection",
            str(tmp_path / "no-existe.json"),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 3
    assert not (tmp_path / "out").exists()


def test_final_validation_requires_run_id_without_selection(tmp_path):
    rc = final_main(
        [
            "--split",
            "validation",
            "--selection",
            str(tmp_path / "no-existe.json"),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out").exists()


def test_validation_split_reads_only_val_never_test(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "crop_id,split,source_image_id\nv1,val,1\nv2,val,2\nt1,test,3\n",
        encoding="utf-8",
    )
    assert MANIFEST_SPLITS["validation"] == "val"
    assert _source_images(manifest, MANIFEST_SPLITS["validation"]) == {"v1": "1", "v2": "2"}
    assert _source_images(manifest, MANIFEST_SPLITS["test"]) == {"t1": "3"}
