from pydantic import BaseModel, Field, model_validator
import json
import os


class FuncParam(BaseModel):
    type: str


class FuncDef(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    parameters: dict[str, FuncParam]
    returns: dict[str, str]


class JsonFile(BaseModel):
    file_input: str
    file_func: str

    @model_validator(mode="after")
    @classmethod
    def check_file_exists(self):
        if not os.path.exists(self.file_input):
            raise OSError(f"[JSON] No access to the file: {self.file_input}")
        if not os.path.exists(self.file_func):
            raise OSError(f"[JSON] No access to the file: {self.file_func}")

    def load_json_files(cls, file_input: str,
                        file_func: str) -> tuple[list[str],
                                                 list[str]]:
        with open(file_input, 'r') as f:
            input = json.load(f)
        with open(file_func, 'r') as f:
            func = json.load(f)
        return input, func
