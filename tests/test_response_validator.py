import pytest
from app.pipelines.response_validator import ResponseValidator


@pytest.fixture
def validator():
    return ResponseValidator()


def test_valid_response_with_sources(validator):
    report = validator.validate(
        response_text="گندم کی زنگ لگنے پر یہ علاج کریں",
        rag_sources=["FAO Wheat Disease Guide"],
        vision_confidence="high",
    )
    assert report.has_rag_source is True
    assert report.low_confidence is False
    assert report.disclaimer == ""


def test_no_sources_adds_disclaimer(validator):
    report = validator.validate(
        response_text="کچھ مشورہ",
        rag_sources=[],
        vision_confidence="high",
    )
    assert report.has_rag_source is False
    assert len(report.disclaimer) > 0
    assert len(report.warnings) > 0


def test_low_vision_confidence_adds_disclaimer(validator):
    report = validator.validate(
        response_text="فصل میں بیماری نظر آتی ہے",
        rag_sources=["Some Source"],
        vision_confidence="low",
    )
    assert report.low_confidence is True
    assert len(report.disclaimer) > 0


def test_none_sources_treated_as_empty(validator):
    report = validator.validate(
        response_text="مشورہ",
        rag_sources=None,
        vision_confidence="high",
    )
    assert report.has_rag_source is False


# --- Phase 4: ungrounded treatment detection ---


def test_dosage_in_rag_text_not_flagged(validator):
    report = validator.validate(
        response_text="Apply 500ml of fungicide per hectare.",
        rag_sources=["doc_1"],
        rag_texts=["Use 500ml of approved fungicide per hectare for wheat rust."],
    )
    assert report.has_ungrounded_treatment is False


def test_dosage_not_in_rag_text_flagged(validator):
    report = validator.validate(
        response_text="Apply 500ml of XYZ chemical per hectare.",
        rag_sources=["doc_1"],
        rag_texts=["Wheat rust can be managed with resistant varieties."],
    )
    assert report.has_ungrounded_treatment is True
    assert any("dosage" in w.lower() for w in report.warnings)


def test_no_dosage_in_response_not_flagged(validator):
    report = validator.validate(
        response_text="فصل کو صاف رکھیں اور بیماری والے پتے ہٹائیں۔",
        rag_sources=["doc_1"],
        rag_texts=["Remove infected leaves to prevent spread."],
    )
    assert report.has_ungrounded_treatment is False


def test_no_rag_texts_skips_dosage_check(validator):
    report = validator.validate(
        response_text="Apply 200ml pesticide.",
        rag_sources=["doc_1"],
        rag_texts=None,
    )
    assert report.has_ungrounded_treatment is False


def test_empty_rag_texts_skips_dosage_check(validator):
    report = validator.validate(
        response_text="Apply 200ml pesticide.",
        rag_sources=["doc_1"],
        rag_texts=[],
    )
    assert report.has_ungrounded_treatment is False
