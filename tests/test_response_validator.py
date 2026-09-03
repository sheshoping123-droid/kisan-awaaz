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
