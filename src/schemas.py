from typing import Literal

from pydantic import BaseModel, Field


class CustomerData(BaseModel):

    gender: Literal["Female", "Male"] = Field(
        ..., description="Customer's gender."
    )
    SeniorCitizen: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer is a senior citizen."
    )
    Partner: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has a partner."
    )
    Dependents: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has dependents."
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=72,
        description=(
            "Number of months the customer has stayed with the company. "
            "Bounded to 0-72 to match the tenure_group buckets the trained pipeline "
            "was fit on (values outside this range have no valid bucket)."
        ),
    )
    PhoneService: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has a phone service."
    )
    MultipleLines: Literal["No", "No phone service", "Yes"] = Field(
        ..., description="Whether the customer has multiple phone lines."
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service provider type."
    )
    OnlineSecurity: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has online security add-on."
    )
    OnlineBackup: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has online backup add-on."
    )
    DeviceProtection: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has device protection add-on."
    )
    TechSupport: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has tech support add-on."
    )
    StreamingTV: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has streaming TV add-on."
    )
    StreamingMovies: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has streaming movies add-on."
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Customer's contract term."
    )
    PaperlessBilling: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has paperless billing."
    )
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., description="Customer's payment method.")
    MonthlyCharges: float = Field(
        ..., ge=0, description="Amount charged to the customer monthly."
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Total amount charged to the customer so far (0.0 for a brand-new customer)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.50,
                "TotalCharges": 0.0,
            }
        }
    }


class PredictionResponse(BaseModel):

    prediction: Literal["No", "Yes"] = Field(
        ..., description="Predicted churn label from the saved model."
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of churn (the 'Yes' class)."
    )
