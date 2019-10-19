import torch
import pickle
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.input_optimization.activation_utilities import ActivationStatistics

weights_path = '/Volumes/Elements/Projekte/Immersions/checkpoints/immersions_scalogram_resnet/_ckpt_epoch_20.ckpt'
tags_path = '/Volumes/Elements/Projekte/Immersions/logs/immersions_scalogram_resnet/version_7/meta_tags.csv'
ranges_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/immersions/immersions/misc/immersions_scalogram_resnet_ranges.p'

model = ContrastivePredictiveSystem.load_from_metrics(weights_path, tags_path, on_gpu=torch.cuda.is_available())
model.freeze()
model.eval()
register = model.activation_register

item_length = model.preprocessing_receptive_field + 63 * 4096
with open(ranges_path, 'rb') as handle:
    activation_ranges = pickle.load(handle)

activation_statistics = ActivationStatistics()

for i in range(10):
    print("batch", i)
    #silence_input = torch.zeros(4, 1, item_length)
    silence_input = torch.randn(16, 1, item_length) * 1e-6
    _ = model(silence_input)
    activations = register.get_activations()
    for key, range in activation_ranges.items():
        activations[key] = activations[key][:, :, :, range[0]:range[1]]
    activation_statistics.add_activation_batch(activations)

t_mean, t_std, e_mean, e_std = activation_statistics.condense_statistics()

result_dict = {'total_mean': t_mean,
               'total_std': t_std,
               'element_mean': e_mean,
               'element_std': e_std}

with open('immersions_scalogram_resnet_silence_statistics.p', 'wb') as handle:
    pickle.dump(result_dict, handle)

pass



