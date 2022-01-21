from .mixer import AbstractMixer, SpecialPrecondition
from .linear import LinearMixer
from .pulay import PulayMixer

def Mixer(scheme = None, **kwargs):
    if scheme == 'Pulay' :
        mixer = PulayMixer(**kwargs)
    elif scheme == 'Linear' :
        mixer = LinearMixer(**kwargs)
    elif scheme is None or scheme.lower().startswith('no'):
        mixer = None
    else :
        raise AttributeError("!!!ERROR : NOT support ", scheme)
    return mixer
