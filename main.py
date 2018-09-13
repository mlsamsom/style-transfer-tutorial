from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

from data.io import load_img
from data.vizualize import show_results
from data.processing import processVGG19, deprocessVGG19

from model.vgg_style import run_style_transfer


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("-s", "--style", help="Path to style image")
    ap.add_argument("-c", "--content", help="Path to content image")
    args = ap.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # load and preprocess images
    style = load_img(args.style)
    style_k = processVGG19(style)
    content = load_img(args.content)
    content_k = processVGG19(content)

    # Run model
    best, best_loss = run_style_transfer(content_k, style_k)

    # show image
    best = deprocessVGG19(best)
    show_results(best, content.astype('uint8'), style.astype('uint8'))


if __name__ == "__main__":
    main()
