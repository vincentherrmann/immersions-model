from unittest import TestCase
from immersions.input_optimization.optimization import Optimization


class TestOptimization(TestCase):
    def test_loading(self):
        weights_path = '/Volumes/Elements/Projekte/Immersions/checkpoints/immersions_scalogram_resnet/_ckpt_epoch_20.ckpt'
        tags_path = '/Volumes/Elements/Projekte/Immersions/logs/immersions_scalogram_resnet/version_7/meta_tags.csv'
        model_shapes_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/immersions_scalogram_resnet_layout_2/immersions_scalogram_resnet_activation_shapes.p'
        ranges_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/immersions_scalogram_resnet_layout_2/immersions_scalogram_resnet_ranges.p'
        noise_statistics_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/immersions_scalogram_resnet_layout_2/immersions_scalogram_resnet_silence_statistics.p'
        data_statistics_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/immersions_scalogram_resnet_layout_2/immersions_scalogram_resnet_data_statistics.p'
        soundclips_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/soundclips_44khz'

        optimization = Optimization(weights_path=weights_path,
                                    tags_path=tags_path,
                                    model_shapes_path=model_shapes_path,
                                    ranges_path=ranges_path,
                                    noise_statistics_path=noise_statistics_path,
                                    data_statistics_path=data_statistics_path,
                                    soundclips_path=soundclips_path)
        optimization.step()
        pass