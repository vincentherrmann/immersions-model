import torch
import pickle
import pytorch_lightning as pl
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.cpc_system_maestro import ContrastivePredictiveSystemMaestro
from immersions.input_optimization.activation_utilities import ActivationStatistics
from immersions.classication_task import ClassificationTaskModel, MaestroClassificationTaskModel
import numpy as np


#model_path = "/home/vincent/Projects/Immersions/models/immersions_house_3/"
#model_path = "/home/vincent/Projects/Immersions/models/immersions_house_4_score_over_timesteps/"
#model_path = "/home/vincent/Projects/Immersions/models/immersions_maestro_3/"
model_path = "/home/vincent/Projects/Immersions/models/immersions_maestro_4_score_over_timesteps/"
checkpoint = model_path + "_clean_checkpoint.ckpt"
#tags = model_path + "meta_tags.csv"
tags = None
#ranges_path = model_path + "activation_ranges.p"
num_hidden_layers = 0

if "maestro" in model_path:
    system = ContrastivePredictiveSystemMaestro
else:
    system = ContrastivePredictiveSystem

if tags is None:
    model = system.load_from_checkpoint(checkpoint)
else:
    model = system.load_from_metrics(checkpoint, tags_csv=tags)
model.cuda()
model.on_gpu = True
model.freeze()
model.eval()
model.batch_size = 64


maestro_training_set_path = '/home/vincent/data/maestro-v2.0.0'
maestro_validation_set_path = '/home/vincent/data/maestro-v2.0.0'

maestro_losses = []
maestro_accuracies = []

for i in range(5):
    task_model = MaestroClassificationTaskModel(cpc_system=model,
                                                task_dataset_path=maestro_validation_set_path,
                                                feature_size=model.ar_model.ar_size,
                                                hidden_layers=num_hidden_layers)
    task_model.calculate_data()

    trainer = pl.Trainer(logger=False, checkpoint_callback=False)
    trainer.fit(task_model)

    maestro_losses.append(trainer.callback_metrics["val_loss"])
    maestro_accuracies.append(trainer.callback_metrics["val_accuracy"])

maestro_losses = np.asarray(maestro_losses)
maestro_accuracies = np.asarray(maestro_accuracies)
maestro_validation_dict = {
    "loss_mean": maestro_losses.mean(),
    "loss_std": maestro_losses.std(),
    "accuracy_mean": maestro_accuracies.mean(),
    "accuracy_std": maestro_accuracies.std()
}


house_task_set_path = '/home/vincent/data/house_data_mp3/test_task'

house_losses = []
house_accuracies = []

for i in range(5):
    task_model = ClassificationTaskModel(cpc_system=model,
                                         task_dataset_path=house_task_set_path,
                                         feature_size=model.ar_model.ar_size,
                                         hidden_layers=num_hidden_layers)
    task_model.calculate_data()

    trainer = pl.Trainer(logger=False, checkpoint_callback=False)
    trainer.fit(task_model)

    house_losses.append(trainer.callback_metrics["val_loss"])
    house_accuracies.append(trainer.callback_metrics["val_accuracy"])

house_losses = np.asarray(house_losses)
house_accuracies = np.asarray(house_accuracies)
house_validation_dict = {
    "loss_mean": house_losses.mean(),
    "loss_std": house_losses.std(),
    "accuracy_mean": house_accuracies.mean(),
    "accuracy_std": house_accuracies.std()
}

print("model:", checkpoint, "classification hiddden layers:", num_hidden_layers)
print("maestro:", maestro_validation_dict)
print("house:", house_validation_dict)

# with attention:
# 1 hidden layers: {'loss': 1.0141394138336182, 'train_loss': 1.0141394138336182, 'val_loss': 0.890221357345581, 'val_accuracy': 0.6720319390296936}
# 0 hidden layers: {'loss': 0.5078184008598328, 'train_loss': 0.5078184008598328, 'val_loss': 1.202830195426941, 'val_accuracy': 0.6035165786743164}

# without attention:
# 0 hidden layers: {'loss': 0.4678333103656769, 'train_loss': 0.4678333103656769, 'val_loss': 0.7114659547805786, 'val_accuracy': 0.7506351470947266}
# 1 hidden layers: {'loss': 0.59615558385849, 'train_loss': 0.59615558385849, 'val_loss': 0.580730676651001, 'val_accuracy': 0.7874148488044739}
pass



