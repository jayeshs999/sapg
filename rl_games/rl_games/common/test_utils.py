import torch
def set_all_to_zero(batch_dict):
    for k in batch_dict:
        if isinstance(batch_dict[k], torch.Tensor):
            batch_dict[k] = torch.zeros_like(batch_dict[k])
    if 'rnn_states' in batch_dict:
        batch_dict['rnn_states'] = [torch.zeros_like(batch_dict['rnn_states'][i]) for i in range(len(batch_dict['rnn_states']))]
    batch_dict['sigmas'] = torch.ones_like(batch_dict['sigmas'])
    return batch_dict

def hash_state_dict(state_dict):
    
    return {k : state_dict[k].sum() if isinstance(state_dict[k], torch.Tensor) else state_dict[k] for k in state_dict}