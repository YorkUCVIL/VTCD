import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'third_party/aot-benchmark/'))
from models.hide_seek.tcow.third_party.aot_benchmark.networks.engines.aot_engine import AOTEngine, AOTInferEngine


def build_engine(name, phase='train', **kwargs):
    if name == 'aotengine':
        if phase == 'train':
            return AOTEngine(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
