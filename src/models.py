from pydantic import BaseModel, Field, model_validator
from typing import Any, Self
import json
import os


class FuncParam(BaseModel):
    type: str


class FuncDef(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    parameters: dict[str, FuncParam]
    returns: dict[str, str]


class PromptInput(BaseModel):
    prompt: str


class JsonFile(BaseModel):
    file_input: str
    file_func: str

    def load_json_files(self) -> tuple[list[dict[str, Any]],
                                       list[dict[str, Any]]]:
        with open(self.file_input, 'r') as f:
            input_data = json.load(f)
        with open(self.file_func, 'r') as f:
            func_data = json.load(f)
        return input_data, func_data

    @model_validator(mode="after")
    def check_file_exists(self) -> Self:
        if not os.path.exists(self.file_input):
            raise OSError(f"[JSON] No access to the file: {self.file_input}")
        if not os.path.exists(self.file_func):
            raise OSError(f"[JSON] No access to the file: {self.file_func}")
        return self
