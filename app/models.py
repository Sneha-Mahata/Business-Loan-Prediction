from pydantic import BaseModel, Field
from typing import Optional, List

class LoanApplication(BaseModel):
    """
    Pydantic model for loan application data
    """
    Selected: int = Field(1, description="Selection indicator")
    ApprovalDate: int = Field(..., description="Approval date in YYYYMMDD format")
    ApprovalFY: int = Field(..., description="Approval fiscal year")
    Term: int = Field(..., description="Loan term in months")
    NoEmp: int = Field(..., description="Number of employees")
    NewExist: float = Field(..., description="Business status: 1.0 for existing, 2.0 for new")
    CreateJob: int = Field(..., description="Number of jobs created")
    RetainedJob: int = Field(..., description="Number of jobs retained")
    DisbursementDate: float = Field(..., description="Disbursement date in YYYYMMDD format")
    DisbursementGross: int = Field(..., description="Gross disbursement amount in INR")
    GrAppv: int = Field(..., description="Gross approved amount in INR")
    New: int = Field(..., description="New business indicator (0 or 1)")
    RealEstate: int = Field(..., description="Real estate secured indicator (0 or 1)")
    TermLoan: int = Field(..., description="Term loan indicator (0 or 1)")
    RevLineCr_Y: int = Field(..., description="Revolving line of credit indicator (0 or 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "Selected": 1,
                "ApprovalDate": 20240101,
                "ApprovalFY": 2024,
                "Term": 240,
                "NoEmp": 45,
                "NewExist": 1.0,
                "CreateJob": 5,
                "RetainedJob": 45,
                "DisbursementDate": 20240115,
                "DisbursementGross": 2125000000,
                "GrAppv": 2125000000,
                "New": 0,
                "RealEstate": 1,
                "TermLoan": 1,
                "RevLineCr_Y": 0
            }
        }

class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response
    """
    default_probability: float = Field(..., description="Probability of loan default")
    default_prediction: bool = Field(..., description="Binary prediction (True for default, False for no default)")
    risk_category: str = Field(..., description="Risk category (Low, Medium, High, Very High)")

class HealthCheckResponse(BaseModel):
    """
    Pydantic model for health check response
    """
    status: str
    model_loaded: bool