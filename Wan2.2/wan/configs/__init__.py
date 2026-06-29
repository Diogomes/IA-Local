# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B
from .wan_s2v_14B import s2v_14B
from .wan_t2v_A14B import t2v_A14B
from .wan_ti2v_5B import ti2v_5B
from .wan_animate_14B import animate_14B

WAN_CONFIGS = {
    't2v-A14B': t2v_A14B,
    'i2v-A14B': i2v_A14B,
    'ti2v-5B': ti2v_5B,
    'animate-14B': animate_14B,
    's2v-14B': s2v_14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704),
    '1024*704': (1024, 704),
    '704*1024': (704, 1024),
    # PATCH (photo2video): resoluções intermediárias (qualidade boa que cabe em
    # GPUs de ~12-16GB sem OOM). 960*544 ≈ 540p, 832*480 ≈ 480p.
    '960*544': (960, 544),
    '544*960': (544, 960),
    # PATCH (photo2video): tamanhos pequenos só para SMOKE TEST em CPU.
    '256*256': (256, 256),
    '384*384': (384, 384),
    '512*512': (512, 512),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
    '1024*704': 1024 * 704,
    '704*1024': 704 * 1024,
    # PATCH (photo2video): áreas intermediárias (qualidade boa, cabe em 12-16GB).
    '960*544': 960 * 544,
    '544*960': 544 * 960,
    # PATCH (photo2video): áreas pequenas para SMOKE TEST em CPU.
    '256*256': 256 * 256,
    '384*384': 384 * 384,
    '512*512': 512 * 512,
}

SUPPORTED_SIZES = {
    't2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'ti2v-5B': ('704*1280', '1280*704', '960*544', '544*960', '832*480',
                '480*832', '256*256', '384*384', '512*512'),
    's2v-14B': ('720*1280', '1280*720', '480*832', '832*480', '1024*704',
                '704*1024', '704*1280', '1280*704'),
    'animate-14B': ('720*1280', '1280*720')
}
