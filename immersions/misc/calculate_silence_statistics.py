import torch
import pickle
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.cpc_training_maestro import ContrastivePredictiveSystemMaestro
from immersions.input_optimization.activation_utilities import ActivationStatistics

model_path = "/home/vincent/Projects/Immersions/models/immersions_maestro_3/"
checkpoint = model_path + "_clean_checkpoint.ckpt"
tags = None #model_path + "meta_tags.csv"
ranges_path = model_path + "activation_ranges.p"

weights = torch.load(checkpoint, map_location='cpu')
save_clean_checkpoint = False
for key in list(weights['state_dict'].keys()):
    if "test_task_model" in key:
        save_clean_checkpoint = True
        del weights['state_dict'][key]
if save_clean_checkpoint:
    checkpoint = model_path + "_clean_checkpoint.ckpt"
    torch.save(weights, checkpoint)

if "maestro" in  model_path:
    system = ContrastivePredictiveSystemMaestro
else:
    system = ContrastivePredictiveSystem

if tags is None:
    model = system.load_from_checkpoint(checkpoint)
else:
    model = system.load_from_metrics(checkpoint, tags)
model.cuda()
model.on_gpu = True
model.freeze()
model.eval()
model.activation_register.active = True
model.model.module.viz_mode = True
register = model.activation_register

item_length = 9 * 44100
with open(ranges_path, 'rb') as handle:
    activation_ranges = pickle.load(handle)

activation_statistics = ActivationStatistics()

for i in range(10):
    print("batch", i)
    #silence_input = torch.zeros(4, 1, item_length)
    silence_input = torch.randn(12, 1, item_length) * 1e-6
    _ = model(silence_input)
    activations = register.get_activations()
    for key, range in activation_ranges.items():
        activations[key] = activations[key][:, :, :, -range:].cpu()
    activation_statistics.add_activation_batch(activations)

t_mean, t_std, e_mean, e_std = activation_statistics.condense_statistics()

result_dict = {'total_mean': t_mean,
               'total_std': t_std,
               'element_mean': e_mean,
               'element_std': e_std}

statistics_path = model_path + "silence_statistics.p"
with open(statistics_path, 'wb') as handle:
    pickle.dump(result_dict, handle)

pass



