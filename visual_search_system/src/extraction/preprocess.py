from torchvision import transforms
from PIL import Image

class Preprocessing:

    def __init__(self, image_size):
        self.image_size = image_size # 64 or 224 depending on model

        self.processing = transforms.Compose(
            [
                # Resize
                transforms.Resize((self.image_size, self.image_size)),
                # Scales and convert to tensor
                transforms.ToTensor(), 
            ]
        )


    def process(self, probe):
        probe = self.processing(probe)
        return probe

# Example 
if __name__ == "__main__":
    image_size = 64
    image_path = "visual_search_system/simclr_resources/probe/Aaron_Sorkin/Aaron_Sorkin_0002.jpg"
    preprocessing = Preprocessing(image_size=image_size)
    probe = Image.open(image_path)
    print(preprocessing.process(probe).shape)