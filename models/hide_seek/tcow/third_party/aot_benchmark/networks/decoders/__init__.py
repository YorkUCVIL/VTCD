from models.hide_seek.tcow.third_party.aot_benchmark.networks.decoders.fpn import FPNSegmentationHead


def build_decoder(name, **kwargs):

    if name == 'fpn':
        return FPNSegmentationHead(**kwargs)
    else:
        raise NotImplementedError
