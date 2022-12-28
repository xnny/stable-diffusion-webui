# -*- coding:utf-8 -*-
#
# User: 'xnny'
# DateTime: 2022-12-27 19:46
import argparse

from modules.textual_inversion.textual_inversion import create_embedding, train_embedding


def add_text_embedding(name: str, num_vectors_per_token: int, overwrite_old: bool = False, init_text: str = '*'):
    create_embedding(
        name, num_vectors_per_token, overwrite_old, init_text
    )


def begin_training(name: str, learn_rate: float, batch_size: int, gradient_step: int, data_root, log_directory,
                   training_width, training_height, steps, shuffle_tags, tag_drop_out, latent_sampling_method,
                   create_image_every, save_embedding_every, template_file, save_image_with_stored_embedding,
                   preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps,
                   preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    train_embedding(
        embedding_name=name,
        learn_rate=learn_rate,
        batch_size=batch_size,
        gradient_step=gradient_step,
        data_root=data_root,
        log_directory=log_directory,
        training_width=training_width,
        training_height=training_height,
        steps=steps,
        shuffle_tags=shuffle_tags,
        tag_drop_out=tag_drop_out,
        latent_sampling_method=latent_sampling_method,
        create_image_every=create_image_every,
        save_embedding_every=save_embedding_every,
        template_file=template_file,
        save_image_with_stored_embedding=save_image_with_stored_embedding,
        preview_from_txt2img=preview_from_txt2img,
        preview_prompt=preview_prompt,
        preview_negative_prompt=preview_negative_prompt,
        preview_steps=preview_steps,
        preview_sampler_index=preview_sampler_index,
        preview_cfg_scale=preview_cfg_scale,
        preview_seed=preview_seed,
        preview_width=preview_width,
        preview_height=preview_height
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', dest='fn', type=str, required=False, default='add_text_embedding',)
    parser.add_argument('--name', dest='name', type=str, required=False, default='test',)
    parser.add_argument('--initialization_text', dest='num_vectors_per_token', type=int, required=False, default=1,)
    parser.add_argument('--overwrite_old', action='store_true', dest='overwrite_old', type=bool, required=False,
                        default=False,)
    parser.add_argument('--num_vectors_per_token', dest='num_vectors_per_token', type=int, required=False, default=1,)

    parser.add_argument('--learn_rate', dest='learn_rate', type=float, required=False, default=0.005,)
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=False, default=1,)
    parser.add_argument('--gradient_step', dest='gradient_step', type=int, required=False, default=1,)
    parser.add_argument('--data_root', dest='data_root', type=str, required=False, default='',)
    parser.add_argument('--log_directory', dest='log_directory', type=str, required=False, default='textual_inversion',)
    parser.add_argument('--training_width', dest='training_width', type=int, required=False, default=512,)
    parser.add_argument('--training_height', dest='training_height', type=int, required=False, default=512,)
    parser.add_argument('--steps', dest='steps', type=int, required=False, default=100000,)
    parser.add_argument('--shuffle_tags', action='store_true', dest='shuffle_tags', type=bool, required=False,
                        default=False,)
    parser.add_argument('--tag_drop_out', dest='tag_drop_out', type=int, required=False, default=0,)
    parser.add_argument('--latent_sampling_method', dest='latent_sampling_method', type=str, required=False,
                        default='once',)
    parser.add_argument('--create_image_every', dest='create_image_every', type=int, required=False, default=500,)
    parser.add_argument('--save_embedding_every', dest='save_embedding_every', type=int, required=False, default=500,)
    parser.add_argument('--template_file', dest='template_file', type=str, required=False,
                        default='/content/stable-diffusion-webui/textual_inversion_templates/style_filewords.txt',)
    parser.add_argument('--save_image_with_stored_embedding', action='store_true',
                        dest='save_image_with_stored_embedding', type=bool, required=False, default=True,)
    parser.add_argument('--preview_from_txt2img', action='store_true', dest='preview_from_txt2img', type=bool,
                        required=False, default=False,)
    parser.add_argument('--preview_prompt', dest='preview_prompt', type=str, required=False, default='',)
    parser.add_argument('--preview_negative_prompt', dest='preview_negative_prompt', type=str, required=False,
                        default='',)
    parser.add_argument('--preview_steps', dest='preview_steps', type=int, required=False, default=20,)
    parser.add_argument('--preview_sampler_index', dest='preview_sampler_index', type=int, required=False, default=0,)
    parser.add_argument('--preview_cfg_scale', dest='preview_cfg_scale', type=float, required=False, default=7,)
    parser.add_argument('--preview_seed', dest='preview_seed', type=int, required=False, default=-1,)
    parser.add_argument('--preview_width', dest='preview_width', type=int, required=False, default=512,)
    parser.add_argument('--preview_height', dest='preview_height', type=int, required=False, default=512,)

    cmd_opts = parser.parse_args()
    fn = cmd_opts.fn
    name = cmd_opts.name
    if fn == 'add_text_embedding':
        init_text = cmd_opts.initialization_text
        overwrite_old = cmd_opts.overwrite_old
        num_vectors_per_token = cmd_opts.num_vectors_per_token
        add_text_embedding(
            name=name,
            num_vectors_per_token=num_vectors_per_token,
            overwrite_old=overwrite_old,
            init_text=init_text
        )
    if fn == 'begin_training':
        # noinspection DuplicatedCode
        learn_rate = cmd_opts.learn_rate
        batch_size = cmd_opts.batch_size
        gradient_step = cmd_opts.gradient_step
        data_root = cmd_opts.data_root
        log_directory = cmd_opts.log_directory
        training_width = cmd_opts.training_width
        training_height = cmd_opts.training_height
        steps = cmd_opts.steps
        shuffle_tags = cmd_opts.shuffle_tags
        tag_drop_out = cmd_opts.tag_drop_out
        latent_sampling_method = cmd_opts.latent_sampling_method
        create_image_every = cmd_opts.create_image_every
        # noinspection DuplicatedCode
        save_embedding_every = cmd_opts.save_embedding_every
        template_file = cmd_opts.template_file
        save_image_with_stored_embedding = cmd_opts.save_image_with_stored_embedding
        preview_from_txt2img = cmd_opts.preview_from_txt2img
        preview_prompt = cmd_opts.preview_prompt
        preview_negative_prompt = cmd_opts.preview_negative_prompt
        preview_steps = cmd_opts.preview_steps
        preview_sampler_index = cmd_opts.preview_sampler_index
        preview_cfg_scale = cmd_opts.preview_cfg_scale
        preview_seed = cmd_opts.preview_seed
        preview_width = cmd_opts.preview_width
        preview_height = cmd_opts.preview_height
        begin_training(
            name=name,
            learn_rate=learn_rate,
            batch_size=batch_size,
            gradient_step=gradient_step,
            data_root=data_root,
            log_directory=log_directory,
            training_width=training_width,
            training_height=training_height,
            steps=steps,
            shuffle_tags=shuffle_tags,
            tag_drop_out=tag_drop_out,
            latent_sampling_method=latent_sampling_method,
            create_image_every=create_image_every,
            save_embedding_every=save_embedding_every,
            template_file=template_file,
            save_image_with_stored_embedding=save_image_with_stored_embedding,
            preview_from_txt2img=preview_from_txt2img,
            preview_prompt=preview_prompt,
            preview_negative_prompt=preview_negative_prompt,
            preview_steps=preview_steps,
            preview_sampler_index=preview_sampler_index,
            preview_cfg_scale=preview_cfg_scale,
            preview_seed=preview_seed,
            preview_width=preview_width,
            preview_height=preview_height
        )


if __name__ == '__main__':
    main()
