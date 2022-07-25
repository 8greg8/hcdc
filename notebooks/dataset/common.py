"""
 Copyright (C) 2021 GORENJE d. o. o. - All Rights Reserved
 
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 
 Written by Gregor Koporec
"""
import random
import matplotlib.pyplot as plt
from rich import print
from collections import Counter
import pandas as pd


def show_rand_img(dset, supercats, imgid=None):
    supercats = supercats if isinstance(supercats, list) else [supercats]
    catids = dset.get_catids(supercats=supercats)

    imgids = dset.get_imgids(catids)
    if imgid is None:
        imgid = random.choice(imgids)
    annids = dset.get_annids(catids, [imgid])
    anns = dset.load_anninfos(annids)
    img = dset.load_annimg(imgid, annids)

    print("#" * 80)
    print(f"imgid: {imgid}")
    print(f"number of anns: {len(annids)}")
    for ann in anns:
        catinfo = dset.load_catinfos([ann["category_id"]])[0]
        if ann["is_occluded"]:
            print(f"{catinfo['name']} is occluded.")
    print("#" * 80)

    plt.imshow(img)
    plt.show()
    return imgid


"""
CATS
"""
MODE = {"cats", "imgs", "anns"}


def check_equal_wp(dset, pdset, mode):
    assert mode in MODE
    # If categories are equal
    cdf = pd.DataFrame.from_dict(getattr(dset, mode).values())
    pdf = pd.DataFrame.from_dict(getattr(pdset, mode).values())
    print(f"{mode} {cdf.equals(pdf)}")


"""
ANNS
"""
WP = {"wholes", "parts"}


def check_catids_in_anns(dset, wp):
    assert wp in WP
    # Check same categories in anns
    # Some categories could be missing and thus you get false
    if wp == "wholes":
        catids = set(dset.get_wholecat_ids())
    elif wp == "parts":
        catids = set(dset.get_partcat_ids())
    anncatids = set(ann["category_id"] for ann in dset.anns.values())

    if catids != anncatids:
        print(f"{wp} False")
        if catids.issuperset(anncatids):
            missing = catids - anncatids
            print(
                        f"Missing {len(missing)} ids in annotations: {missing}"
                    )
            catinfos = dset.load_catinfos(missing)
            print(catinfos)
            
        else:
            print(f"Missing in cats: {anncatids - catids}")
    else:
        print(f"{wp} True")


def check_imgids_in_anns(dset, wp):
    assert wp in WP

    if wp == "wholes":
        # Check same imgids in anns
        wimgids = set(dset.imgids)
        wannimgids = set(ann["image_id"] for ann in dset.anns.values())
        print(f"Wholes {wimgids == wannimgids}")
    elif wp == "parts":
        pimgids = set(dset.imgids)
        pannimgids = set(ann["image_id"] for ann in dset.anns.values())
        if dset:
            if pimgids != pannimgids:
                print(f"Parts False")
                if pimgids.issuperset(pannimgids):
                    missing = pimgids - pannimgids
                    print(
                        f"Missing {len(missing)} ids in annotations: {missing}"
                    )
                    # imginfos = dset.load_imginfos(list(missing))
                    # print(imginfos)
                    if len(missing) < 15:
                        for mid in missing:
                            # Print original image name
                            imginfo = dset.load_imginfos([mid])[0]
                            print(f"orig name: {imginfo['image_url']}")
                            # Show ann occlusions
                            annids = dset.get_annids(imgids=[mid])
                            anninfos = dset.load_anninfos(annids)
                            print(
                                f"is_occluded: {[anninfo['is_occluded'] for anninfo in anninfos]}"
                            )

                            # Plot annotations
                            wimg = dset.load_annimg(mid)
                            pimg = dset.load_annimg(mid)
                            fig, axs = plt.subplots(1, 2)
                            axs[0].imshow(wimg)
                            axs[1].imshow(pimg)
                            plt.show()
                else:
                    print(f"Missing in imgs: {pannimgids - pimgids}")
            else:
                print("Parts True")


def check_ocln_anns(dset):
    occluded = [
        ann["category_id"] for ann in dset.anns.values() if ann["is_occluded"]
    ]
    all = [ann["category_id"] for ann in dset.anns.values()]
    allcounter = Counter(all)
    oclncounter = Counter(occluded)
    print(f"all:{allcounter}")
    print(f"ocln:{oclncounter}")
    allcounter.subtract(oclncounter)
    print(f"non-ocln: {allcounter}")
