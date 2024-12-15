from model.segformer.segformer_predict import SegformerPredict
from util.config.config_reader import *

def predict():
    config = ConfigReader()
    cuda_enable = config.get_config('predict', 'cuda', ConfigFileType.BOOL)
    mode = config.get_config('predict', 'mode', ConfigFileType.INT)
    model_path = config.get_config('predict', 'model_path', ConfigFileType.STR)
    num_classes = config.get_config('predict', 'num_classes', ConfigFileType.INT)
    auto_colored = config.get_config('predict', 'auto_colored', ConfigFileType.BOOL)
    feature_extraction_fun = config.get_config('predict', 'feature_extraction_fun', ConfigFileType.STR)
    input_shape_str = config.get_config('predict', 'input_shape', ConfigFileType.STR)
    input_shape = list(map(int, input_shape_str.split(',')))
    mix_type = config.get_config('predict', 'mix_type', ConfigFileType.INT)

    if mode == 0:
        args = (cuda_enable, model_path, num_classes, auto_colored,
                [], feature_extraction_fun, input_shape, mix_type)
        segformer = SegformerPredict(args)

