import math
import random
import numpy as np
from PIL import Image


def augment(image: Image, seed: int = None) -> Image:
    if seed is not None:
        random.seed(seed)

    # Rotation
    rotation_angle = random.randint(-180, 180)
    rotated_image = image.rotate(rotation_angle, expand=True)

    # Cropping non-image area after rotation. For the math,
    # see: https://stackoverflow.com/questions/21346670/cropping-rotated-image-with-same-aspect-ratio
    aspect_ratio = image.size[0] / image.size[1]
    rotated_aspect_ratio = rotated_image.size[0] / rotated_image.size[1]
    angle = math.fabs(rotation_angle) * math.pi / 180

    if aspect_ratio < 1:
        total_height = float(image.size[0]) / rotated_aspect_ratio
    else:
        total_height = float(image.size[1])

    h = total_height / (aspect_ratio * math.fabs(math.sin(angle)) + math.fabs(math.cos(angle)))
    w = h * aspect_ratio

    left = (rotated_image.size[0] - w) / 2
    upper = (rotated_image.size[1] - h) / 2
    right = left + w
    lower = upper + h
    cropped_image = rotated_image.crop((left, upper, right, lower))

    # Horizontal flipping
    flip_prob = random.random()
    if flip_prob < 0.5:
        flipped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flipped_image = cropped_image

    # Vertical flipping
    flip_prob = random.random()
    if flip_prob < 0.5:
        flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)

    return flipped_image


def remove_transparency(image: Image) -> Image:
    if image.mode in ('RGBA', 'RGBa', 'LA', 'La', 'PA', 'P'):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image = image.convert('RGB')
    return image


def resize_crop(image: Image, width: int, height: int) -> Image:
    original_aspect_ratio = image.width / image.height
    target_aspect_ratio = width / height

    if original_aspect_ratio > target_aspect_ratio:
        # Crop horizontally
        new_width = int(image.height * target_aspect_ratio)
        left = (image.width - new_width) // 2
        upper = 0
        right = left + new_width
        lower = image.height
    else:
        # Crop vertically
        new_height = int(image.width / target_aspect_ratio)
        left = 0
        upper = (image.height - new_height) // 2
        right = image.width
        lower = upper + new_height

    cropped_image = image.crop((left, upper, right, lower))
    resized_image = cropped_image.resize((width, height), Image.Resampling.LANCZOS)

    return resized_image


def normalize_pixels(image: Image) -> Image:
    image_array = np.array(image)
    normalized_image_array = image_array / 255.0  # Normalize pixel values to the range [0, 1]
    return Image.fromarray((normalized_image_array * 255).astype(np.uint8))
