import torch
import pickle
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.cpc_system_maestro import ContrastivePredictiveSystemMaestro
from immersions.input_optimization.activation_utilities import ActivationStatistics

model_path = "/Volumes/Elements/Projekte/Immersions/models/immersions_maestro_new"
#training_set_path = '/Volumes/Elements/Datasets/Immersions/house_data_mp3/training'
training_set_path = "/Volumes/Elements/Datasets/maestro-v2.0.0"
#validation_set_path = '/Volumes/Elements/Datasets/Immersions/house_data_mp3/validation'
validation_set_path = "/Volumes/Elements/Datasets/maestro-v2.0.0"

weights_path = model_path + "/_clean_checkpoint.ckpt"
tags_path = model_path + "/version_0/meta_tags.csv"
ranges_path = model_path + "/activation_ranges.p"

model = ContrastivePredictiveSystemMaestro.load_from_metrics(weights_path,
                                                             tags_path,
                                                             on_gpu=torch.cuda.is_available())
model.freeze()
model.eval()

model.hparams.training_set_path = training_set_path
model.hparams.validation_set_path = validation_set_path
model.batch_size = 16
model.activation_register.active = True
model.setup_datasets()
model.model.module.viz_mode = True
register = model.activation_register

item_length = 9 * 44100
with open(ranges_path, 'rb') as handle:
    activation_ranges = pickle.load(handle)

activation_statistics = ActivationStatistics()

dataloader = model.val_dataloader

for i, batch in enumerate(dataloader):
    if i >= 50:
        break
    print("batch", i)
    batch = batch[0][:, :, :item_length]
    _ = model(batch)
    activations = register.get_activations()
    for key, range in activation_ranges.items():
        activations[key] = activations[key][:, :, :, -range:]
    activation_statistics.add_activation_batch(activations)

t_mean, t_std, e_mean, e_std = activation_statistics.condense_statistics()

result_dict = {'total_mean': t_mean,
               'total_std': t_std,
               'element_mean': e_mean,
               'element_std': e_std}

statistics_path = model_path + "/data_statistics.p"
with open(statistics_path, 'wb') as handle:
    pickle.dump(result_dict, handle)

pass



