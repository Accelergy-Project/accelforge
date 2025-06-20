from pydantic import TypeAdapter, ValidationError
import islpy as isl

class PyISLSet(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, _):
        from pydantic_core import core_schema

        def validate(value, _info):
            try:
                # Attempt to parse the string as an ISL set
                return isl.Set(value)
            except Exception as e:
                raise ValueError(f"Invalid ISL set: {e}")

        return core_schema.no_info_plain_validator_function(validate)

ISLSetAdapter = TypeAdapter(PyISLSet)

# Usage example
try:
    s = ISLSetAdapter.validate_python("{ [i, j] : 0 <= i < 10 and j = i + 1 }")
    print(s)  # This is an islpy.Set instance
except ValidationError as e:
    print(e)