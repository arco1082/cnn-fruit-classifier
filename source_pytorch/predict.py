# import libraries
import os
import numpy as np
import torch
from six import BytesIO

# import model from model.py, by name
from model import FruitClassifier

# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'


# Provided model load function
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitClassifier(114)

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


# Provided input data loading
def input_fn(input_data, content_type):
    print('Deserializing the input data.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np_array = encoders.decode(input_data, content_type)
    tensor = torch.FloatTensor(np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    return tensor.to(device)

# Provided output data handling
def output_fn(prediction, accept):
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


# Provided predict function
def predict_fn(data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    out_np = output.cpu().detach().numpy()
    out_label = out_np.round()

    return out_label

