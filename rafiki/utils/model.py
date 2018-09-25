import dill
import os

def serialize_model(model_inst):
    model_bytes = dill.dumps(model_inst)
    return model_bytes

def serialize_model_to_file(model_inst, out_file_path):
    model_serialized = serialize_model(model_inst)

    with open(out_file_path, 'wb') as f:
        f.write(model_serialized)

    print('Serialized model file saved at {}'.format(out_file_path))

def unserialize_model(model_class_bytes):
    model_inst = dill.loads(model_class_bytes)

    return model_inst
