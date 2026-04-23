import copy
import torch


def aggregate(global_model, client_weights, client_sizes):
    """
    Federated averaging: weighted average by number of local samples.
    Used by both FedAvg and FedProx (aggregation step is identical).
    """
    total_samples = sum(client_sizes)
    avg_state = copy.deepcopy(client_weights[0])

    for key in avg_state:
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
        for i, state in enumerate(client_weights):
            weight = client_sizes[i] / total_samples
            avg_state[key] += state[key].float() * weight

    global_model.load_state_dict(avg_state)
    return global_model
