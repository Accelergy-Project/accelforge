from typing import Any, Type

import islpy as isl

from pydantic import BaseModel, ValidationError
from pydantic_core import core_schema

class ISLAdapter:
    def __init__(self, isl_type: Type):
        self.isl_type = isl_type

    def __get_pydantic_core_schema__(self, _source, _handler):
        isl_type = self.isl_type  # capture for closure

        def validate(value: Any):
            if not isinstance(value, str):
                raise TypeError('Value must be a string')
            try:
                return isl_type(value)
            except Exception as e:
                raise ValueError(f'Invalid input for {isl_type.__name__}: {e}')
        return core_schema.no_info_plain_validator_function(validate)

ISLMap = ISLAdapter(isl.Map)
ISLSpace = ISLAdapter(isl.Space)
ISLSet = ISLAdapter(isl.Set)