import dill

from .BaseModel import BaseModel

class InvalidModelException(Exception):
    pass

def serialize_model(model_inst):
    # TODO: Validate class definition
    if not isinstance(model_inst, BaseModel):
        raise InvalidModelException()

    model_bytes = dill.dumps(model_inst)
    return model_bytes

def unserialize_model(model_class_bytes):
    model_inst = dill.loads(model_class_bytes)

    return model_inst
