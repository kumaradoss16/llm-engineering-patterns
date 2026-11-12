"""Pydantic schemas for technical information extraction."""

import ipaddress

"""
BaseModel -	Creates Pydantic data models
Field -	Defines constraints and default values
field_validator - Creates custom field validation logic
"""
from pydantic import BaseModel, Field, field_validator

# Extracts technical information form Logs.
class LogExtraction(BaseModel):
    service: str | None = Field(default=None, max_length=100)
    port: int | None = Field(default=None, ge=1, le=65535)
    error_code: str | None = Field(default=None, max_length=100)
    ip_address: str | None = Field(default=None, max_length=45)
    version: str | None = Field(default=None, max_length=50)
    event: str | None = Field(default=None, max_length=200)

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, value: str | None) -> str | None:
        if value is None:
            return None

        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {value!r}") from exc

        return value


class SoftwareExtraction(BaseModel):
    software_name: str | None = Field(default=None, max_length=100)
    version: str | None = Field(default=None, max_length=50)
    configuration_key: str | None = Field(default=None, max_length=200)
    configuration_value: str | None = Field(default=None, max_length=500)