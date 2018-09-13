# Style transfer tutorial

## Description

Adaptation of style transfer tutorial from tensorflow website

## System Requirements

The requirements.txt has a non-gpu tensorflow. If you have CUDA and a gpu it is recommended to use tensorflow-gpu

On a laptop with a 2 GB GPU the process takes ~5min, this would be much slower CPU only.

## USAGE

```console
pip install -r requirements.txt
python main.py -s test_images/style_test.jpg -c test_images.content_test.jpg
```