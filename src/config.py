"""Shared settings used by training, evaluation and serving."""

IMG_SIZE = 224          # input resolution expected by ImageNet-pretrained models
RESIZE_SIZE = 256       # eval: resize the short side to 256, then center-crop 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 37
SEED = 42

# The Oxford-IIIT Pet dataset has 12 cat breeds and 25 dog breeds.
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
}


def species_of(breed: str) -> str:
    return "cat" if breed in CAT_BREEDS else "dog"
