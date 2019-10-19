from unittest import TestCase
import pytorch_lightning as pl
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.classication_task import ClassificationTaskModel

class TestClassificationTaskModel(TestCase):
    def test_classification_task_remote(self):
        weights_path = '/home/idivinci3005/experiments/checkpoints/_ckpt_epoch_20.ckpt'
        tags_path = '/home/idivinci3005/experiments/logs/immersions_scalogram_resnet/version_7/meta_tags.csv'
        task_dataset_path = '/home/idivinci3005/data/immersions/small_test_task'
        cpc_system = ContrastivePredictiveSystem.load_from_metrics(weights_path, tags_path, on_gpu=False)
        cpc_system.hparams.batch_size = 4
        task_system = ClassificationTaskModel(cpc_system, task_dataset_path)
        task_system.calculate_data()
        trainer = pl.Trainer()
        trainer.fit(task_system)

