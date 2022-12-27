# -*- coding:utf-8 -*-
#
# User: 'xnny'
# DateTime: 2022-12-27 19:46

from modules.textual_inversion.textual_inversion import create_embedding


def add_text_embedding(name: str, num_vectors_per_token: int, overwrite_old: bool = False, init_text: str = '*'):
    create_embedding(
        name, num_vectors_per_token, overwrite_old, init_text
    )

