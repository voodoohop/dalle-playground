from cog import BasePredictor, Path, Input, File
import sys, os
import uuid
sys.path.insert(1, './backend')
from backend.consts import ModelSize
from backend.dalle_model import DalleModel
from io import BytesIO
import typing

dalle_model = None
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        dalle_version = ModelSize.MINI
        self.model = DalleModel(dalle_version)
        print("setup complete")


    def predict(self,
                prompt: str = Input(description="Image prompt"),
                num: int = Input(description="Number of images to generate", default=1)
                ) -> typing.Iterator[Path]:
        """Run a single prediction on the model"""
        import time
        start_time = time.time()

        print("generating images")
        generated_imgs = self.model.generate_images(prompt, num)
        image = None
        for img in generated_imgs:
            img_name = uuid.uuid4()
            img.save(f'{img_name}.png')
            yield Path(f'{img_name}.png')
        execution_time = (time.time() - start_time)
        print('Execution time in seconds: ' + str(execution_time))
        return Path('output.png')