from pathlib import Path
from typing import Any, List

from matplotlib import lines, patches, pyplot as plt  # type: ignore
from PIL import Image  # type: ignore


def draw_bboxes(img: Any, bboxes: List[List[int]], labels: List[int], scores: List[float], classes: List[str]) -> None:
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    assert len(bboxes) == len(labels) == len(scores)
    for bbox, label, score in zip(bboxes, labels, scores):
        props = dict(capstyle='butt', facecolor=colors[label])
        ax.text(bbox[0], bbox[1], f'{score:.3f}', va='top', ha='left', bbox=props)
        rect = patches.Rectangle(bbox[:2],
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1],
                                 linewidth=1.5,
                                 edgecolor=colors[label],
                                 facecolor='none')
        ax.add_patch(rect)

    custom_lines = [lines.Line2D([0], [0], color=c, lw=4) for c in colors]
    plt.legend(custom_lines, classes)
    plt.show()


def main() -> None:
    img_path = Path(__file__).resolve().parents[1] / 'data' / 'color' / 'Parrots.bmp'
    img = Image.open(img_path, mode='r').convert('RGB')

    bboxes = [[0, 50, 90, 254], [80, 80, 254, 254]]
    labels = [0, 1]
    scores = [0.87, 0.23]
    classes = ['bird1', 'bird2']
    draw_bboxes(img, bboxes, labels, scores, classes)


if __name__ == '__main__':
    main()
