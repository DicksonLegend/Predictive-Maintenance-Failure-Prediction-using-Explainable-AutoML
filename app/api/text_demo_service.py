"""Text Demo API Service wrapper."""
from app.api.schemas import TextExplainRequest, TextExplainResponse
from src.xai.text_explainer import text_explainer_service


def explain_maintenance_text(request: TextExplainRequest) -> TextExplainResponse:
    """Wrapper function to analyze maintenance notes text."""
    return text_explainer_service.explain(request.text)
