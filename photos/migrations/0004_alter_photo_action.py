# Generated by Django 3.2.2 on 2022-05-25 06:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('photos', '0003_alter_photo_action'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='action',
            field=models.CharField(choices=[('NO_FILTER', 'NO_FILTER'), ('GRAY', 'GRAY'), ('XYZ', 'XYZ'), ('YCrCb', 'YCrCb'), ('HSV', 'HSV'), ('HLS', 'HLS'), ('CIElab', 'CIElab'), ('CIEluv', 'CIEluv'), ('BINARY', 'BINARY'), ('BINARY_INV', 'BINARY_INV'), ('TRUNC', 'TRUNC'), ('TOZERO', 'TOZERO'), ('TOZERO_INV', 'TOZERO_INV'), ('OTSU', 'OTSU'), ('BLURRED', 'BLURRED'), ('BOX_FILTER', 'BOX_FILTER'), ('GAUSSIANBLUR', 'GAUSSIANBLUR'), ('MEDIANBLUR', 'MEDIANBLUR'), ('BILATERALFILTER', 'BILATERALFILTER'), ('EROSION', 'EROSION'), ('DILATION', 'DILATION'), ('MORPH_OPEN', 'MORPH_OPEN'), ('MORPH_CLOSE', 'MORPH_CLOSE'), ('MORPH_TOPHAT', 'MORPH_TOPHAT'), ('MORPH_BLACKHAT', 'MORPH_BLACKHAT'), ('SOBEL', 'SOBEL'), ('SCHARR', 'SCHARR'), ('LAPLACIAN', 'LAPLACIAN'), ('CANNY', 'CANNY')], max_length=50, null=True),
        ),
    ]
