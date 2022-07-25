"""
 Copyright (C) 2020 GORENJE d. o. o. - All Rights Reserved
 
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 
 Written by Gregor Koporec
"""
from pycocotools.coco import COCO
from argparse import ArgumentParser
from pathlib import Path
from grcore.coco.file_io import load_coco, save_coco
from grcore.coco import ann
from fridgecore.coco.occ import find_breakpoint
from grcore.common.utils import FlipDict
import copy


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--coco", type=str, metavar="<coco>", help="COCO json file",
    )
    parser.add_argument(
        "--coco2", type=str, metavar="<coco2>", help="Second COCO json file",
    )
    parser.add_argument(
        "--occ",
        choices=["iou", "occupancy"],
        default="iou",
        type=str,
        metavar="<occ>",
        help="Occupancy type",
    )
    return parser


def main(args):
    # Load instance segmentation coco dataset
    old_coco = load_coco(args.coco)
    old_coco2 = load_coco(args.coco2) if args.coco2 else None

    # Create new coco dataset
    new_coco = COCO()
    if "info" in old_coco.dataset:
        new_coco.dataset["info"] = copy.deepcopy(old_coco.dataset["info"])
    if "licenses" in old_coco.dataset:
        new_coco.dataset["licenses"] = copy.deepcopy(
            old_coco.dataset["licenses"]
        )

    # Create images
    if old_coco2 is not None:
        imginfos = copy.deepcopy(old_coco.dataset["images"]) + copy.deepcopy(
            old_coco2.dataset["images"]
        )
        imgs = []
        for id_, imginfo in enumerate(imginfos, 1):
            imginfo["id"] = id_
            imgs.append(imginfo)
        new_coco.dataset["images"] = imgs
    else:
        new_coco.dataset["images"] = copy.deepcopy(old_coco.dataset["images"])

    # Create new categories
    cats = [0.1, 0.2, 0.3, None]
    id2cat = FlipDict({id_: cat for id_, cat in enumerate(cats, 1)})
    catinfos = []
    for id_, cat in id2cat.items():
        catinfos.append(ann.build_cat_info(id_, str(cat)))
    new_coco.dataset["categories"] = catinfos

    # Create new annotations
    anns = []
    for id_, imginfo in enumerate(new_coco.dataset["images"], 1):
        cat = find_breakpoint(imginfo[args.occ], cats)
        catid = id2cat.flip[cat]
        annot = ann.build_ann_info(id_, imginfo["id"], catid)
        anns.append(annot)
    new_coco.dataset["annotations"] = anns

    path = Path(args.coco)
    old_stem = path.stem.split("_")[0]
    suffix = path.suffix

    # Save file
    if args.coco2:
        path2 = Path(args.coco2)
        old_stem2 = path2.stem.split("_")[0]
        path = path.with_name(f"{old_stem}{old_stem2}_{args.occ}{suffix}")
    else:
        path = path.with_name(f"{old_stem}_{args.occ}{suffix}")
    save_coco(new_coco, path)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
