from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    model_version: str
