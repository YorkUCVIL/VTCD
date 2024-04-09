from models.hide_seek.tcow.third_party.aot_benchmark.networks.models.aot import AOT


def build_vos_model(name, cfg, **kwargs):

    if name == 'aot':
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
