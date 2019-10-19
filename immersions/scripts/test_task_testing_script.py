import pytorch_lightning as pl
import torch
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.classication_task import ClassificationTaskModel



weights_path = '/home/idivinci3005/experiments/checkpoints/_ckpt_epoch_4.ckpt'
tags_path = '/home/idivinci3005/experiments/logs/immersions_scalogram_resnet/version_7/meta_tags.csv'
task_dataset_path = '/home/idivinci3005/data/immersions/test_task'
if torch.cuda.is_available():
    cpc_system = ContrastivePredictiveSystem.load_from_metrics(weights_path, tags_path, on_gpu=True)
    cpc_system.cuda()
else:
    cpc_system = ContrastivePredictiveSystem.load_from_metrics(weights_path, tags_path, on_gpu=False)


hparams = cpc_system.hparams
cpc_system = ContrastivePredictiveSystem(hparams)
if torch.cuda.is_available():
    cpc_system.cuda()

cpc_system.freeze()
cpc_system.eval()
#cpc_system.hparams.batch_size = 4
task_system = ClassificationTaskModel(cpc_system, task_dataset_path)
task_system.calculate_data()
trainer = pl.Trainer(max_nb_epochs=20)
trainer.fit(task_system)
test_task_acc = trainer.tng_tqdm_dic['avg_val_accuracy']
print("accuracy:", test_task_acc * 100., "%")
pass

# _ckpt_epoch_5.ckpt  - accuracy: 79.94791865348816 %
# _ckpt_epoch_10.ckpt - accuracy: 79.296875 %
# _ckpt_epoch_20.ckpt - accuracy: 79.81770634651184 %

# small test task network:
# untrained: 10.15625 %
# epoch  4: 54.55729365348816 %
# epoch  5: 64.71354365348816 %
# epoch 10: 67.70833134651184 %
# epoch 15: 66.53645634651184 %
# epoch 20: 67.31770634651184 %


