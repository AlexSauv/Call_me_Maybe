from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Any, Self
import json
import os

VALID_TYPES = ['string', 'number', 'integer', 'boolean', 'null']


class FieldType(BaseModel):
    """Model definition for returns and parameters types"""
    type: str

    @field_validator("type", mode="after")
    def type_checker(cls, value: str) -> str:
        if value not in VALID_TYPES:
            raise ValueError(f"[TYPE] {value} type not found.")
        return value


class FuncDef(BaseModel):
    """Model represnetation of a function """
    name: str = Field(default="")
    description: str = Field(default="")
    parameters: dict[str, FieldType]
    returns: FieldType


class FuncResult(BaseModel):
    """Model representation of the expected output"""
    prompt: str
    name: str
    parameters: dict[str, Any]


class PromptInput(BaseModel):
    """Model representing a single input prompt."""
    prompt: str


class JsonFile(BaseModel):
    """Helper to validate and load input JSON files safely."""
    file_input: str
    file_func: str

    @model_validator(mode="after")
    def check_file_exists(self) -> Self:
        if not os.path.exists(self.file_input):
            raise OSError(f"[JSON] No access to the file: {self.file_input}")
        if not os.path.exists(self.file_func):
            raise OSError(f"[JSON] No access to the file: {self.file_func}")
        return self

    def load_json_files(self) -> tuple[list[PromptInput], list[FuncDef]]:
        """Reads input files and returns validated Pydantic models."""
        try:
            with open(self.file_input, 'r', encoding='utf-8') as f:
                raw_inputs = json.load(f)
                input_data = [PromptInput(**item) for item in raw_inputs]
            with open(self.file_func, 'r', encoding='utf-8') as f:
                raw_funcs = json.load(f)
                func_data = [FuncDef(**func) for func in raw_funcs]
            return input_data, func_data
        except json.JSONDecodeError as e:
            raise ValueError(f"[JSON] Invalid JSON format in input files: {e}")
