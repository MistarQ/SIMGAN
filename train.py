from options.config import get_arguments
from utils.manipulate import *
from models.SIMGAN import *
import utils.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--root', help='input image dir', default='datas/apple')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    Gs2 = []
    Zs2 = []
    reals2 = []
    NoiseAmp2 = []
    dir2save = functions.generate_dir2save(opt)
    # add domainC
    Gs3 = []
    Zs3 = []
    reals3 = []
    NoiseAmp3 = []

    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    realA, realB, realC = functions.read_three_domains(opt)
    functions.adjust_scales2image(realA, opt)
    train(opt, Gs, Zs, reals, NoiseAmp, Gs2, Zs2, reals2, NoiseAmp2, Gs3, Zs3, reals3, NoiseAmp3)
    SIMGAN_generate(Gs, Zs, reals, NoiseAmp, Gs2, Zs2, reals2, NoiseAmp2, Gs3, Zs3, reals3, NoiseAmp3, opt)
