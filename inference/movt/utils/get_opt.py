import ast
from argparse import Namespace
from os.path import join as pjoin

from .word_vectorizer import POS_enumerator


def parse_opt_value(value):
    """Parse a value written by the project's plain-text option format."""

    value = value.strip()
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _set_missing(opt, **defaults):
    for key, value in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)


def get_opt(opt_path, device, **overrides):
    """Load an ``opt.txt`` file without discarding checkpoint metadata."""

    opt = Namespace()
    with open(opt_path, "r") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("---") or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            setattr(opt, key, parse_opt_value(value))

    vars(opt).update(overrides)
    _set_missing(opt, checkpoints_dir="./checkpoints", which_epoch="finest")

    dataset_defaults = {
        "t2m": {
            "data_root": "./dataset/HumanML3D/",
            "motion_dir": "./dataset/HumanML3D/new_joint_vecs",
            "text_dir": "./dataset/HumanML3D/texts",
            "joints_num": 22,
            "dim_pose": 263,
        },
        "mpp": {
            "data_root": "./dataset/Motion-X++/",
            "motion_dir": "./dataset/Motion-X++/vector_263",
            "text_dir": "./dataset/Motion-X++/text",
            "joints_num": 22,
            "dim_pose": 66,
        },
        "kit": {
            "data_root": "./dataset/KIT-ML/",
            "motion_dir": "./dataset/KIT-ML/new_joint_vecs",
            "text_dir": "./dataset/KIT-ML/texts",
            "joints_num": 21,
            "dim_pose": 251,
        },
    }
    if not hasattr(opt, "dataset_name") or opt.dataset_name not in dataset_defaults:
        dataset_name = getattr(opt, "dataset_name", None)
        raise KeyError(f"Dataset not recognized: {dataset_name}")
    _set_missing(opt, **dataset_defaults[opt.dataset_name])
    _set_missing(
        opt,
        max_motion_length=196,
        max_motion_frame=196,
        max_motion_token=55,
        unit_length=4,
    )

    if not hasattr(opt, "name"):
        raise ValueError(f"Missing 'name' in option file: {opt_path}")
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.meta_dir = pjoin(opt.save_root, "meta")
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device
    return opt
