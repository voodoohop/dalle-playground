from cog import BasePredictor, Path, Input, File
import sys
sys.path.insert(1, './backend')
from backend.consts import ModelSize
from backend.dalle_model import DalleModel
from io import BytesIO


dalle_model = None
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        dalle_version = ModelSize.MINI
        self.model = DalleModel(dalle_version)

    def predict(self,
                prompt: str = Input(description="Image prompt"),
                num: int = Input(description="Number of images to generate", default=1)
                ) -> Path:
        """Run a single prediction on the model"""
        # ... pre-processing ...
        generated_imgs = self.model.generate_images(prompt, num)
        image = None
        for img in generated_imgs:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            image = img
        return File(image)