import torch

from model.ann import Ann
from model.lenet import LeNet
import torch.optim as optim


def get_model(model_name, input_size, in_channels, num_classes, device=torch.device('cpu'), weight_path=None):
    if model_name == 'ann':
        model = Ann(input_size=(input_size**2)*in_channels, num_classes=num_classes)
    elif model_name == 'lenet':
        model = LeNet(input_size=input_size, in_channels=in_channels, num_classes=num_classes)
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        print('Weight loaded.')
    return model.to(device)


# Load a MNIST or Fashion MNIST trained model, then convert it to be able to be trained on Caltech 101/Caltech 256
def get_pretrained_model_and_convert(model_name, input_size, num_classes, device=torch.device('cpu'), weight_path=None):
    trained_model = get_model(model_name, input_size=input_size, in_channels=1, num_classes=10, device=device, weight_path=weight_path)
    if model_name == 'ann':
        model = Ann(input_size=(input_size**2)*3, num_classes=num_classes)
        model.hidden_layers[0].weight.data = torch.cat([trained_model.hidden_layers[0].weight.data] * 3, dim=1)
        model.hidden_layers[0].bias.data = trained_model.hidden_layers[0].bias.data.clone()
        model.hidden_layers[2].weight.data = trained_model.hidden_layers[2].weight.data.clone()
        model.hidden_layers[2].bias.data = trained_model.hidden_layers[2].bias.data.clone()
        learning_rates = {
            'hidden_layers.0.weight': 0.001,  # First linear layer's weight
            'hidden_layers.0.bias': 0.001,     # First linear layer's bias
            'classifier.0.weight': 0.001,      # Classifier's weight
            'classifier.0.bias': 0.001,        # Classifier's bias
            'hidden_layers.2.weight': 1e-9,   # Second linear layer's weight
            'hidden_layers.2.bias': 1e-9,     # Second linear layer's bias
        }
    elif model_name == 'lenet':
        model = LeNet(input_size=input_size, in_channels=3, num_classes=num_classes)
        for name, param in trained_model.features.state_dict().items():
            if 'features.0.weight' in name:
                model.state_dict()[name] = param.repeat(1, 3, 1, 1)
            else:
                model.state_dict()[name] = param
        learning_rates = {
            'features.0.weight': 0.001,  # First linear layer's weight
            'features.0.bias': 0.001,     # First linear layer's bias
            'features.3.weight': 1e-9,      # Classifier's weight
            'features.3.bias': 1e-9,        # Classifier's bias
            'classifier.0.weight': 0.001,
            'classifier.0.bias': 0.001,
            'classifier.2.weight': 0.001,
            'classifier.2.bias': 0.001,
            'classifier.4.weight': 0.001,
            'classifier.4.bias': 0.001,
        }

    parameters_to_optimize = []
    for name, param in model.named_parameters():
        if name in learning_rates:
            parameters_to_optimize.append({'params': param, 'lr': learning_rates[name]})
    return model.to(device), optim.Adam(parameters_to_optimize)
