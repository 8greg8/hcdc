"""
 Copyright (C) 2020 GORENJE d. o. o. - All Rights Reserved
 
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 
 Written by Gregor Koporec
"""


from typing import AsyncGenerator
from fridge_dataset.config import get_cfg_defaults, setup_cfg
from fridge_dataset import (
    coco_dataset,
    pascalparts_dataset,
    op3d_dataset,
    occlude_voc,
    occlude_coco,
    aug_coco,
)
from argparse import ArgumentParser
import ray
import joblib
from grcore.common.timing import Timer


def get_parser():
    parser = ArgumentParser("Fridge dataset")
    parser.add_argument(
        "--cfg", required=True, metavar="<cfg>", help="Configuration file."
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Parallel execution",
    )
    return parser


def setup(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    setup_cfg(cfg)
    cfg.freeze()
    print(cfg)
    return cfg


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup(args)

    timer = Timer()
    timer.start()

    if args.parallel:
        # ray.init()
        pass

    if cfg.DATASET.TYPE == "yolo":
        pass
    elif cfg.DATASET.TYPE == "coco":
        coco_dataset.write_dataset(args, cfg)
    elif cfg.DATASET.TYPE == "export":
        pass
    elif cfg.DATASET.TYPE == "pascalparts":
        pascalparts_dataset.write_dataset(args, cfg)
    elif cfg.DATASET.TYPE == "op3d":
        op3d_dataset.write_dataset(args, cfg)
    elif cfg.DATASET.TYPE == "ocln_voc":
        occlude_voc.write_dataset(args, cfg)
    elif cfg.DATASET.TYPE == "ocln_coco":
        occlude_coco.write_dataset(args, cfg)
    elif cfg.DATASET.TYPE == "aug_coco":
        aug_coco.write_dataset(args, cfg)
    else:
        print(f"Invalid dataset name: {cfg.DATASET.TYPE}")
    print(f"Elapsed time: {timer.elapsed_time()}")
