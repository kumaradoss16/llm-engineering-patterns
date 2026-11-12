"""Pydantic schemas for classification outputs."""

from pydantic import BaseModel, Field

ALLOWED_TICKET_LABELS = {"Network", "Hardware", "Software", "Security"}
ALLOWED_ERROR_CATEGORIES = {
    "Configuration",
    "Connectivity",
    "Resource",
    "Permission",
    "Unknown"
}

class TicketClassification(BaseModel):
    label: str = Field(pattern="^(Network|Hardware|Software|Security)$")
    reasoning: str = Field(default="", max_length=500)

class ErrorClassification(BaseModel):
    category: str = Field(pattern="^(Configuration|Connectivity|Resource|Permission|Unknown)$")
    reasoning: str = Field(default="", max_length=500)