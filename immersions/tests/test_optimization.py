from unittest import TestCase
from immersions.input_optimization.optimization import Optimization
from immersions_control_app.settings_dicts import maestro_new

class TestOptimization(TestCase):
    def test_loading(self):
        settings = maestro_new

        optimization = Optimization(weights_path=settings["weights_path"],
                                    tags_path=settings["tags_path"],
                                    model_shapes_path=settings["activations_shapes_path"],
                                    ranges_path=settings["ranges_path"],
                                    noise_statistics_path=settings["noise_statistics_path"],
                                    data_statistics_path=settings["data_statistics_path"],
                                    soundclips_path=settings["soundclips_path"],
                                    dev='cuda:0')

        for i in range(5):
            optimization.step()
        pass