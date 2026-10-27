"""Image Demo API Service."""
from app.api.schemas import ImageDemoResponse
from src.xai.image_explainer import get_image_demo_response


def get_image_demo_items() -> ImageDemoResponse:
    """Returns image explainability demonstration items."""
    return get_image_demo_response()
