from copy import copy
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont  # type: ignore

import matplotlib.pyplot as plt  # type: ignore


def draw_bboxes(img: Image.Image, bboxes: List[List[int]], labels: List[str], scores: List[float],
                label2color: Dict[str, str]) -> None:
    annotated_img = copy(img)

    draw = ImageDraw.Draw(annotated_img)
    font = ImageFont.load_default()

    assert len(bboxes) == len(labels) == len(scores)
    for bbox, label, score in zip(bboxes, labels, scores):
        text = f'{label}: {int(100 * score)}%'
        text_size = font.getsize(text.upper())
        PAD = 3
        text_location = [bbox[0] + PAD, bbox[1] - text_size[1]]
        textbox_location = [bbox[0], text_location[1], bbox[0] + text_size[0] + PAD * 2, bbox[1]]

        draw.rectangle(xy=bbox, outline=label2color[label])
        draw.rectangle(xy=[l + 1 for l in bbox], outline=label2color[label])
        draw.rectangle(xy=textbox_location, fill=label2color[label])
        draw.text(xy=text_location, text=text.upper(), fill='white', font=font)

    plt.imshow(annotated_img)
    plt.show()


def main() -> None:
    img_path = Path(__file__).resolve().parents[1] / 'data' / 'color' / 'Parrots.bmp'
    img = Image.open(img_path, mode='r').convert('RGB')

    bboxes = [[0, 50, 90, 254], [80, 80, 254, 254]]
    labels = ['bird', 'bird2']
    scores = [0.87, 0.23]
    label2color = dict(bird='green', bird2='blue')

    draw_bboxes(img, bboxes, labels, scores, label2color)


if __name__ == '__main__':
    main()
