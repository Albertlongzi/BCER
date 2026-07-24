import torch
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


def generate_uniform_from_range(lb, ub, size=None):
    return np.random.random(size=size) * (ub - lb) + lb


def bSSFP_readout(M0, T1, T2, FA=35):
    """
    bSSFP imaging

    :param M0: Initial magnetization.
    :param T1: T1 map.
    :param T2: T2 map.
    :param FA: Flip angle in DEGREES.
    :return: Image readout.
    """
    cos_fa = np.cos(np.deg2rad(FA))
    sig = M0 / (1 + cos_fa + (1 - cos_fa) * T1 / np.clip(T2, 1e-3, None))
    return sig


def spin_echo_readout(M0, T1, T2, TR=1e3, TE=15):
    """
    Spin echo imaging.

    :param M0: Initial magnetization.
    :param T1: T1 map.
    :param T2: T2 map.
    :param TR: Repetition time.
    :param TE: Echo time
    :return: Image readout.
    """
    E1 = np.exp(-TR / np.clip(T1, 1e-3, None))
    E2 = np.exp(-TE / np.clip(T2, 1e-3, None))
    sig = M0 * (1. - E1) * E2
    return sig


def gradient_echo_readout(M0, T1, T2, FA=15, TR=50, TE=3):
    """
    Gradient echo imaging.

    :param M0: Initial magnetization.
    :param T1: T1 map.
    :param T2: T2 map.
    :param FA: Flip angle in DEGREES!
    :param TR: Repetition time.
    :param TE: Echo time
    :return: Image readout.
    """
    cos_fa = np.cos(np.deg2rad(FA))
    E1 = np.exp(-TR / np.clip(T1, 1e-3, None))
    E2 = np.exp(-TE / np.clip(T2, 1e-3, None))
    sig = M0 * (1. - E1) / (1 - cos_fa * E1) * E2
    return sig


def extract_blood_map(ssfp, seg):
    """
    Extract fuzzy blood map from SSFP cine image and its segmentation. Only support 2D operations!

    :param ssfp: (H, W)
    :param seg: (H, W)
    :return: left ventricle map, right ventricle map, blood map
    """
    # ensure ssfp has a minimum of 0
    ssfp = ssfp - ssfp.min()
    right_ventricle = seg == 1
    left_ventricle = seg > 1
    heart = left_ventricle | right_ventricle
    heart_signal = ssfp[heart]

    # intensity based segmentation: the brighter the signal is, the more likely it is blood.
    blood = np.zeros_like(ssfp)
    q10, q99 = np.percentile(heart_signal, (10, 99))
    heart_signal = (heart_signal - q10) / (q99 - q10)
    heart_signal = np.clip(heart_signal, 0., 1.)
    blood[heart] = heart_signal
    left_ventricle_blood = left_ventricle * blood
    right_ventricle_blood = right_ventricle * blood
    return left_ventricle_blood, right_ventricle_blood, blood


def blood_flow_decay(ssfp, seg, decay_duration: float = 3.):
    """
    Heuristic black blood imaging preparation. It's a bit hard to explain how it works.
    Only support 2D operations!

    :param ssfp: (H, W)
    :param seg: (H, W)
    :param decay_duration: should be in range (1, 20).
    :return: Decay factor of shape (H, W).
    """

    # ensure ssfp has a minimum of 0
    ssfp = ssfp - ssfp.min()
    heart = seg > 0
    heart_signal = ssfp[heart]

    decay_const = (ssfp - heart_signal.min()) * heart  # myocardium, background -> 0, blood -> large value
    decay_const = decay_const.max() - decay_const  # blood -> 0, the rest -> large value
    decay_max = np.percentile(decay_const[heart], 90)  # normalize
    decay_const = np.clip(decay_const / decay_max, 0., 1.) + 1
    decay = np.exp(-decay_duration / decay_const)  # signal decay factor
    decay = (decay - decay.min()) / (decay.max() - decay.min())  # normalize
    return decay



def triple_inversion_recovery_black_blood(ssfp, seg, M0, T1, T2,
                                          heuristic_decay=None, STIR=False,
                                          TIb=0.7, TIf=0.7, T1fat=600, T2fat=150):
    """
    Triple inversion recovery for black-blood imaging with fat suppression.

    :param ssfp: (H, W)
    :param seg: (H, W)
    :param M0: Initial PD.
    :param T1: T1 map.
    :param T2: T2 map.
    :param heuristic_decay: Heuristic decay map.
    :param STIR: Fat suppression option.
    :param TIb: Inversion time in Blood T1 unit, default is 0.7 (approx. ln(2)).
    :param TIf: Inversion time in Fat T1 unit, default is 0.7 (approx. ln(2)).
    :param T1fat: Fat T1 threshold for fat segmentation.
    :param T2fat: Fat T2 threshold for fat segmentation.
    :return: BB-prepared M0.
    """
    T1 = np.clip(T1, 1., None)
    if heuristic_decay is not None:
        M_prepared = M0 * heuristic_decay
    else:
        _, _, blood = extract_blood_map(ssfp, seg)
        T1blood = np.median(T1[blood > 0.5])
        M_flow = M0 * (1 - 2 * np.exp(-TIb * T1blood / T1))
        M_steady = M0
        M_DIR = np.abs(M_steady * (1. - blood) + M_flow * blood)
        M_prepared = M_DIR

    if STIR:
        # perform a third IR for fat suppression
        fat = (T1 < T1fat) & (T2 > T2fat)
        if fat.sum() > 0:
            T1fat = np.median(T1[fat])
            M_prepared = np.abs(M_prepared * (1 - 2 * np.exp(-TIf * T1fat / T1)))

    return M_prepared


def gadolinium_flow(T1, flow_map, concentration_factor: float = 0.5, r1=4.5):
    """
    Simulate T1 variation with Gd. in blood.

    :param T1: Native T1 map.
    :param flow_map: Relative concentration map of Gd.
    :param concentration_factor: Concentration scale.
    :param r1: Relaxation rate of Gd.
    :return:
    """
    T1 = np.clip(T1, 1, None)
    concentration = concentration_factor * flow_map
    T1new = 1 / (concentration * r1 + 1e3 / T1) * 1e3
    return T1new


def saturation_recovery(M0, T1, TI: float = 100):
    """
    Saturation recovery preparation.

    :param M0: Magnetization before SR.
    :param T1: T1 map.
    :param TI: Inversion time.
    :return: SR: Magnetization after SR.
    """
    SR = M0 * (1. - np.exp(-TI / T1))
    return SR


def inversion_recovery(M0, T1, TI: float = 100):
    """
    Inversion recovery preparation.

    :param M0: Magnetization before SR.
    :param T1: T1 map.
    :param TI: Inversion time.
    :return: SR: Magnetization after SR.
    """
    IR = M0 * (1. - 2 * np.exp(-TI / T1))
    return IR


def molli_signal_simu(M0, T1, T2, fa, Tinv, magnitude=True):
    """
    MOLLI signal equation.

    :param M0:  Init magnetization.
    :param T1:  T1 map.
    :param T2:  T2 map.
    :param fa:  flip angle in degrees.
    :param Tinv: Inversion time.
    :param magnitude: If magnitude image should be returned (phase insensitive).
    :return:
    """
    fa = np.deg2rad(fa)
    T1 = np.clip(T1, 1, None)
    T2 = np.clip(T2, 1, None)
    cos_fa = np.cos(fa)
    ratio = T1 / T2
    steady_state = M0 / (1 + cos_fa + (1 - cos_fa) * ratio)
    inv_factor = 1 + np.sin(fa * 0.5) / np.sin(fa) * (ratio * (1 - np.cos(fa)) + 1 + np.cos(fa))
    t1app_inv = 1 / T1 * np.cos(fa * 0.5) ** 2 + 1 / T2 * np.sin(fa * 0.5) ** 2
    t1app = 1 / t1app_inv
    signal = steady_state * (1 - inv_factor * np.exp(-Tinv / t1app))
    if magnitude:
        signal = np.abs(signal)
    return signal


class SplitDataKeyTransform(AbstractTransform):

    def __init__(self, data_key="data"):
        """
        Data is 4-Channel, (SSFP, M0, T1, T2)

        :param data_key:
        """
        self.data_key = data_key

    def __call__(self, **data_dict):
        # data = data_dict[self.data_key]
        ssfp = data_dict[self.data_key][:, [0]]
        phys = data_dict[self.data_key][:, 1:]
        data_dict[self.data_key] = ssfp
        data_dict["phys"] = phys
        return data_dict


class BlackBloodImagingAugmentation(AbstractTransform):
    def __init__(self, data_key='data',
                 p: float = 1.0,
                 phys_key='phys',
                 readouts=("se", "bssfp", "gre")):
        """
        Black blood imaging simulation. There are three types:
            - Direct decay of bSSFP image
            - Double inversion preparation
            - Triple inversion preparation.
        Note that all augmentation parameters are fixed and not configurable.

        :param data_key:
        :param phys_key:
        :param readouts:
        """
        self.data_key = data_key
        self.phys_key = phys_key
        self.p = p
        self.readouts = readouts

    def __call__(self, **data_dict):
        ssfp = data_dict[self.data_key]
        seg = data_dict.get("seg", None)
        phys = data_dict[self.phys_key]

        for ind in range(ssfp.shape[0]):
            p = generate_uniform_from_range(0., 1.)
            if p > self.p:
                continue
            ssfp_ind = ssfp[ind, 0]
            ssfp_ind = ssfp_ind - ssfp_ind.min()
            seg_ind = seg[ind, 0]
            if seg_ind.max() < 1:
                continue
            phys_ind = phys[ind, :]
            M0, T1, T2 = tuple(phys_ind[c] for c in range(3))

            # get decay map
            Tdecay = generate_uniform_from_range(1., 12.)
            decay = blood_flow_decay(ssfp_ind, seg_ind, decay_duration=Tdecay)

            # 1. preparation
            prep = np.random.choice(('direct', 'dir', 'tir'))
            if prep == 'direct':
                M0 = ssfp_ind * decay
            elif prep == 'dir':
                M0 = triple_inversion_recovery_black_blood(ssfp_ind, seg_ind,
                                                           M0, T1, T2, heuristic_decay=decay)
            else:
                TIf = generate_uniform_from_range(0.5, 0.8)
                M0 = triple_inversion_recovery_black_blood(ssfp_ind, seg_ind,
                                                           M0, T1, T2, heuristic_decay=decay,
                                                           STIR=True,
                                                           TIf=TIf)
            # 2. readout
            if not prep == 'direct':
                readout = np.random.choice(('bssfp', 'gre', 'tse'))
                if readout == 'bssfp':
                    fa = generate_uniform_from_range(8, 45)
                    gen_image = bSSFP_readout(M0, T1, T2, FA=fa)
                elif readout == 'gre':
                    fa = generate_uniform_from_range(8, 45)
                    TR = generate_uniform_from_range(20, 200)
                    TE = generate_uniform_from_range(0, 10)
                    gen_image = gradient_echo_readout(M0, T1, T2, fa, TR, TE)
                else:
                    TR = 2000
                    TE = generate_uniform_from_range(0, 50)
                    gen_image = spin_echo_readout(M0, T1, T2, TR, TE)
            else:
                gen_image = M0

            # renormalize
            gen_image_valid = gen_image[seg_ind > -1]
            stats_mu, stats_sig = np.mean(gen_image_valid), np.std(gen_image_valid)
            gen_image = (gen_image - stats_mu) / stats_sig
            gen_image[seg_ind < 0] = 0

            # replace original image
            ssfp[ind, 0] = gen_image

        data_dict["data"] = ssfp
        return data_dict


class PerfusionImagingAugmentation(AbstractTransform):
    def __init__(self, data_key='data',
                 p: float = 0.7,
                 phys_key='phys',
                 readouts=("se", "bssfp", "gre")):
        """
        Perfusion-oriented augmentation.
            - MOLLI for low contrast simulation.
            - Gd. SR + GRE/bSSFP

        :param data_key:
        :param phys_key:
        :param readouts:
        """
        self.data_key = data_key
        self.phys_key = phys_key
        self.p = p
        self.readouts = readouts

    def __call__(self, **data_dict):
        ssfp = data_dict[self.data_key]
        seg = data_dict.get("seg", None)
        phys = data_dict[self.phys_key]

        for ind in range(ssfp.shape[0]):
            p = generate_uniform_from_range(0., 1.)
            if p > self.p:
                continue
            ssfp_ind = ssfp[ind, 0]
            ssfp_ind = ssfp_ind - ssfp_ind.min()
            seg_ind = seg[ind, 0]
            phys_ind = phys[ind, :]
            M0, T1, T2 = tuple(phys_ind[c] for c in range(3))

            perfusion_enhance = generate_uniform_from_range(0., 1.) < 0.5
            if perfusion_enhance and (seg_ind.max() > 1):
                # get T1 after Gd. perfusion.
                lv_factor = generate_uniform_from_range(0.05, 2.0)
                rv_factor = generate_uniform_from_range(0.05, 2.0)
                TI = generate_uniform_from_range(20, 200)
                left_ventricle_blood, right_ventricle_blood, blood = extract_blood_map(ssfp_ind, seg_ind)
                T1new = gadolinium_flow(T1, left_ventricle_blood, concentration_factor=lv_factor, r1=4.5)
                T1new = gadolinium_flow(T1new, right_ventricle_blood, concentration_factor=rv_factor, r1=4.5)
                M0 = saturation_recovery(M0, T1new, TI=TI)
                T1 = T1new

            readout = np.random.choice(("molli", "gre", "bssfp"))
            if readout == 'molli':
                Ti = generate_uniform_from_range(180, np.percentile(T1, 99) * 1.2)
                fa = generate_uniform_from_range(8, 35)
                gen_image = molli_signal_simu(M0, T1, T2, fa, Ti, magnitude=True)
            elif readout == 'gre':
                TR = generate_uniform_from_range(10, 300)
                TE = generate_uniform_from_range(5, 20)
                FA = generate_uniform_from_range(8, 35)
                gen_image = gradient_echo_readout(M0, T1, T2, FA, TR, TE)
            else:
                FA = generate_uniform_from_range(20, 50)
                gen_image = bSSFP_readout(M0, T1, T2, FA)

            # renormalize
            gen_image_valid = gen_image[seg_ind > -1]
            stats_mu, stats_sig = np.mean(gen_image_valid), np.std(gen_image_valid)
            gen_image = (gen_image - stats_mu) / stats_sig
            gen_image[seg_ind < 0] = 0

            # replace original image
            ssfp[ind, 0] = gen_image

        data_dict["data"] = ssfp
        return data_dict

