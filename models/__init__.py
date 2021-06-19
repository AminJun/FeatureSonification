from models.utils import resnet18_audio, resnet50_audio, resnet18_untrained
from models.utils import resnet50_untrained, z_net, vae, decoder, encoder
from models.utils import jasper, greedy_decoder, jasper_augs
from models.jasper import CTCLossNM
from models.z_net import ZNet, ZNetLoss
from models.vae import VAELoss, VAE
from models.hifi import get_hifi_gan

audio_architectures = {'resnet18': resnet18_untrained, 'resnet50': resnet50_untrained, 'z_net': ZNet}
audio_models = {'resnet18': resnet18_audio, 'resnet50': resnet50_audio, 'jasper': jasper, 'z_net': z_net,
                'vae': vae, 'encoder': encoder, 'decoder': decoder}
