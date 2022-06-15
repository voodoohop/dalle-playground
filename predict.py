from cog import BasePredictor, Path, Input, File
import sys
import uuid
import typing
import time
import os
import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from functools import partial
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
import wandb

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # https://github.com/saharmor/dalle-playground/issues/14#issuecomment-1147849318
os.environ["WANDB_SILENT"] = "true"
wandb.init(anonymous="must")

from dalle_mini import DalleBartProcessor

import random

seed = random.randint(0, 2**32 - 1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dalle_model = None
class Predictor(BasePredictor):

    model_name = ""

    def load_dalle(self, model):
        if model == "MINI":
            DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
        elif model == "MEGA":
            DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
        else:
            DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
        print(f'Loading Dalle Model: {DALLE_MODEL}')
        # Load dalle-mini
        model, params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=None, dtype=jnp.float16, _do_init=False
        )
        self.processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=None)
        print(f'Replicating parameters')
        params = replicate(params)
        self.model = model
        self.params = params
        self.model_name = model

    def setup(self):
        print(os.popen("nvidia-smi").read())
        """Load the model into memory to make running multiple predictions efficient"""

        # VQGAN model
        VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
        VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
        print(f'Local Devices: {jax.local_device_count()}')

        self.load_dalle("MINI")
        self.load_dalle("MEGA")

        print(f'Loading VQGAN')
        # Load VQGAN
        vqgan, vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )

        vqgan_params = replicate(vqgan_params)

        self.vqgan_params = vqgan_params
        self.key = jax.random.PRNGKey(seed)
        self.vqgan = vqgan
        print(f'Setup complete')

    def predict(self,
                prompt: str = Input(description="Image prompt"),
                num: int = Input(description="Number of images to generate", default=1, ge=0,le=20),
                model_size: str = Input(description="Size of the model", default="MINI", choices=["MINI", "MEGA", "MEGA_FULL"])
                ) -> typing.List[Path]:
        print(os.popen("nvidia-smi").read())
        # model inference
        start_time = time.time()
        print("loading model")
        self.load_dalle(model_size)
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

        print("Tokenizing")
        tokenized_prompt = self.processor([prompt])
        print("Replicating")
        tokenized_prompt = replicate(tokenized_prompt)


        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0

        print("Generating images")
        for i in range(max(num // jax.device_count(), 1)):
            # get a new key
            key, subkey = jax.random.split(self.key)
            # generate images
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
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
            for decoded_img in decoded_images:
                print(f'saving image {i}')
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                img_name = uuid.uuid4()
                img.save(f'{img_name}.png')
                print(f'image {i} saved to {img_name}.png')
                yield Path(f'{img_name}.png')
            print(f'took {time.time() - start_time}')
        return Path('output.png')





