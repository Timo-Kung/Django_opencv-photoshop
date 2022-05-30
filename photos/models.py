from django.db import models
from .utils import get_filtered_image
from PIL import Image
import numpy as np
from io import BytesIO
from django.core.files.base import ContentFile

# Create your models here.
ACTION_CHOICES = (
    ('NO_FILTER', 'NO_FILTER'),
    ('GRAY', 'GRAY'),
    ('XYZ', 'XYZ'),
    ('YCrCb', 'YCrCb'),
    ('HSV', 'HSV'),
    ('HLS', 'HLS'),
    ('CIElab', 'CIElab'),
    ('CIEluv', 'CIEluv'),
    ('BINARY', 'BINARY'),
    ('BINARY_INV', 'BINARY_INV'),
    ('TRUNC', 'TRUNC'),
    ('TOZERO', 'TOZERO'),
    ('TOZERO_INV', 'TOZERO_INV'),
    ('OTSU', 'OTSU'),
    ('BLURRED', 'BLURRED'),
    ('BOX_FILTER', 'BOX_FILTER'),
    ('GAUSSIANBLUR', 'GAUSSIANBLUR'),
    ('MEDIANBLUR', 'MEDIANBLUR'),
    ('BILATERALFILTER', 'BILATERALFILTER'),
    ('EROSION', 'EROSION'),
    ('DILATION', 'DILATION'),
    ('MORPH_OPEN', 'MORPH_OPEN'),
    ('MORPH_CLOSE', 'MORPH_CLOSE'),
    ('MORPH_TOPHAT', 'MORPH_TOPHAT'),
    ('MORPH_BLACKHAT', 'MORPH_BLACKHAT'),
    ('SOBEL', 'SOBEL'),
    ('SCHARR', 'SCHARR'),
    ('LAPLACIAN', 'LAPLACIAN'),
    ('CANNY', 'CANNY'),

)


class Photo(models.Model):
    name = models.CharField(max_length=100)
    action = models.CharField(max_length=50, choices=ACTION_CHOICES, null=True)
    description = models.TextField()
    image = models.ImageField(upload_to='images')
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)

    def save(self, *args, **kwargs):

        # open image
        pil_img = Image.open(self.image)

        # convert the image to array and do some processing
        cv_img = np.array(pil_img)
        img = get_filtered_image(cv_img, self.action)

        # convert back to pil image
        im_pil = Image.fromarray(img)

        # save
        buffer = BytesIO()
        im_pil.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)

        super().save(*args, **kwargs)
