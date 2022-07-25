"""
 Copyright (C) 2021 GORENJE d. o. o. - All Rights Reserved
 
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 
 Written by Gregor Koporec
"""
from pycocotools.coco import COCO
from argparse import ArgumentParser
from pathlib import Path
from grcore.coco.file_io import load_coco, save_coco
import copy


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--path", type=str, metavar="<coco>", help="Path to annotation files",
    )

    return parser


def main(path):
    # Load instance segmentation coco dataset
    old_coco = load_coco(path)

    # Create new coco dataset
    new_coco = COCO()
    if "info" in old_coco.dataset:
        new_coco.dataset["info"] = copy.deepcopy(old_coco.dataset["info"])

    new_coco.dataset["images"] = copy.deepcopy(old_coco.dataset["images"])
    if "licenses" in old_coco.dataset:
        new_coco.dataset["licenses"] = copy.deepcopy(
            old_coco.dataset["licenses"]
        )
    new_coco.dataset["categories"] = copy.deepcopy(
        old_coco.dataset["categories"]
    )

    # For each annotation add is_occluded and occluders
    new_anns = []
    for id_, ann in old_coco.anns.items():
        ann["is_occluded"] = False
        ann["occluders"] = []
        new_anns.append(ann)
    new_coco.dataset["annotations"] = new_anns

    print(f"Saving results to: {str(path)}")
    save_coco(new_coco, path)


if __name__ == "__main__":
    args = get_parser().parse_args()
    for child in Path(args.path).iterdir():
        if child.is_file() and child.suffix == ".json":
            main(child)
