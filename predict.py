import os
import sys
import time
import typing
import uuid
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from cog import BasePredictor, File, Input, Path
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from PIL import Image
from tqdm.notebook import trange
from transformers import CLIPProcessor, FlaxCLIPModel
from vqgan_jax.modeling_flax_vqgan import VQModel

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # https://github.com/saharmor/dalle-playground/issues/14#issuecomment-1147849318
wandb.init(anonymous="must")

import random

from dalle_mini import DalleBartProcessor

print("jax devices", jax.devices())
seed = random.randint(0, 2**32 - 1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dalle_model = None
class Predictor(BasePredictor):

    model_name = ""

    def load_dalle(self, model):
        if model == "MINI":
            DALLE_MODEL = "/artifacts/mini-1:v0"
        elif model == "MEGA":
            DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
        elif model == "MEGA_FULL":
            DALLE_MODEL = "dalle-mini/dalle-mini/mega-1:latest"
        else:
            DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
        start_time = time.time()
        print(f'Loading Dalle Model: {DALLE_MODEL}')
        print(os.popen("nvidia-smi").read())
        # Load dalle-mini
        self.model, self.params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=None, dtype=jnp.float16, _do_init=False
        )
        print(f'Loading Dalle Processor: {DALLE_MODEL}')
        self.processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=None)
        print(f'Replicating parameters')
        self.params  = replicate(self.params )
        self.model_name = model
        print(f'model loaded in {time.time() - start_time}')

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print(os.popen("nvidia-smi").read())

        # VQGAN model
        VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
        VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
        print(f'Local Devices: {jax.local_device_count()}')

        self.load_dalle("MEGA")

        print(f'Loading VQGAN')
        # Load VQGAN
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )

        self.vqgan_params = replicate(self.vqgan_params)
        self.key = jax.random.PRNGKey(seed)
        print(f'Setup complete')

    def predict(self,
                prompt: str = Input(description="Image prompt"),
                num: int = Input(description="Number of images to generate", default=1, ge=1,le=20),
                model_size: str = Input(description="Size of the model", default="MEGA", choices=["MEGA"])
                ) -> typing.List[Path]:
        print(os.popen("nvidia-smi").read())
        # model inference

        @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
        def p_generate(
                tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
        ):
            return self.model.generate(
                **tokenized_prompt,
                prng_key=key,
                params=params,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                condition_scale=condition_scale,
            )

        # decode image
        @partial(jax.pmap, axis_name="batch")
        def p_decode(indices, params):
            return self.vqgan.decode_code(indices, params=params)
        """Run a single prediction on the model"""


        start_time = time.time()
        print(f'Tokenizing at {time.time() - start_time}')
        tokenized_prompt = self.processor([prompt])
        print(f'Replicating at {time.time() - start_time}')
        tokenized_prompt = replicate(tokenized_prompt)


        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0
        print(f'Generating images at {time.time() - start_time}')
        for i in range(max(num // jax.device_count(), 1)):
            # create a random key
            seed = random.randint(0, 2 ** 32 - 1)
            key = jax.random.PRNGKey(seed)
            # generate images
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(key),
                self.params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            print(f'encoding image {i}')
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            print(f'encoded image {i}')
            # decode images
            decoded_images = p_decode(encoded_images, self.vqgan_params)
            print(f'decoding image {i}')
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            print(f'decoded image {i}')
            all_images = []
            for decoded_img in decoded_images:
                print(f'saving image {i}')
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                img_filename = f'/outputs/{uuid.uuid4()}.png'
                img.save(img_filename)
                print(f'image {i} saved to {img_filename}')
                all_images.append(Path(img_filename))
                yield Path(img_filename)
            print(f'took {time.time() - start_time}')
        return all_images





