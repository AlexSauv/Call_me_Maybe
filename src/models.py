from pydantic import BaseModel, Field, model_validator, ValidationError, field_validator
from typing import Any, Self
import json
import os

VALID_TYPES = ['string', 'number', 'integer', 'boolean', 'null']


class FieldType(BaseModel):
    type: str

    @field_validator("type", mode="after")
    def type_checker(cls, value: str) -> str:
        if value not in VALID_TYPES:
            raise ValueError(f"[TYPE] {value} type not found.")
        return value


def parse_and_validate_output(raw_generation: str, model_class: type[BaseModel]) -> dict:
    try:
        data = json.loads(raw_generation)
        validated_data = model_class(**data)
        return validated_data.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Failed to parse LLM output: {e}")


class FuncResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


class FuncDef(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    parameters: dict[str, FieldType]
    returns: FieldType


class PromptInput(BaseModel):
    prompt: str


class JsonFile(BaseModel):
    file_input: str
    file_func: str

    def load_json_files(self) -> tuple[list[dict[str, Any]],
                                       list[dict[str, Any]]]:
        with open(self.file_input, 'r', encoding='utf-8') as f:
            input_data = [PromptInput(**item) for item in json.load(f)]
        with open(self.file_func, 'r', encoding='utf-8') as f:
            func_data = [FuncDef(**func) for func in json.load(f)]
        return input_data, func_data

    @model_validator(mode="after")
    def check_file_exists(self) -> Self:
        if not os.path.exists(self.file_input):
            raise OSError(f"[JSON] No access to the file: {self.file_input}")
        if not os.path.exists(self.file_func):
            raise OSError(f"[JSON] No access to the file: {self.file_func}")
        return self
