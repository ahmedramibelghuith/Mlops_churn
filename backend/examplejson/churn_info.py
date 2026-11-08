from pydantic import BaseModel, Field
from typing import Literal

class ChurnPredictionModel(BaseModel):
    Age: int = Field(..., ge=0, example=22)
    Gender: Literal["Female", "Male"] = Field(..., example="Female")
    Tenure: int = Field(..., ge=0, example=25)
    Usage_Frequency: int = Field(..., ge=0, alias="Usage Frequency", example=14)  # Alias for space
    Support_Calls: int = Field(..., ge=0, alias="Support Calls", example=4)  # Alias for space
    Payment_Delay: int = Field(..., ge=0, alias="Payment Delay", example=27)  # Alias for space
    Subscription_Type: Literal["Basic", "Standard", "Premium"] = Field(..., alias="Subscription Type", example="Basic")  # Alias for space
    Contract_Length: Literal["Monthly", "Quarterly", "Annual"] = Field(..., alias="Contract Length", example="Monthly")
    Total_Spend: float = Field(..., ge=0, alias="Total Spend", example=598)
    Last_Interaction: int = Field(..., ge=0, alias="Last Interaction", example=9)

    class Config:
        populate_by_name = True
        json_schema_extra = {
        "example": {
                "Age": 22,
                "Gender": "Female",
                "Tenure": 25,
                "Usage Frequency": 14,
                "Support Calls": 4,
                "Payment Delay": 27,
                "Subscription Type": "Basic",
                "Contract Length": "Monthly",
                "Total Spend": 598,
                "Last Interaction": 9,
            }
        }
