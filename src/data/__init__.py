"""API pública do módulo de dados do PA3."""
from .reader import (
    BUDGETS,
    IMAGES_DIR,
    LABELS_DIR,
    SPLITS,
    check_data_layout,
    get_class_names,
    get_image_path,
    get_label_to_name,
    get_name_to_label,
    get_num_classes,
    labeled_dataframe,
    list_image_ids,
    load_budget,
    load_classes,
    load_label_csv,
    load_label_map,
    load_test_public,
    load_val,
    normalize_labels,
    split_for_label_csv,
)
from .datasets import (
    LabeledImageDataset,
    MultiViewImageDataset,
    UnlabeledImageDataset,
)
from . import transforms  # noqa: F401  (permite `from src.data import transforms`)
from .transforms import (
    eval_tf,
    eval_tf_full,
    fixmatch_strong,
    fixmatch_weak,
    simclr_aug,
    simclr_aug_no_crop,
    supervised_eval,
    supervised_train,
    train_tf,
)
from .prompts import (
    DEFAULT_TEMPLATES,
    SCIENTIFIC_NAMES,
    SYNONYMS,
    build_class_prompts,
    humanize,
    load_class_names,
)
from .pseudo import PseudoLabelDataset

__all__ = [
    # constantes
    "IMAGES_DIR", "LABELS_DIR", "SPLITS", "BUDGETS",
    # leitura
    "load_classes", "get_class_names", "get_num_classes",
    "get_label_to_name", "get_name_to_label",
    "load_label_csv", "load_budget", "load_val", "load_test_public",
    "load_label_map", "split_for_label_csv", "normalize_labels",
    "get_image_path", "list_image_ids", "labeled_dataframe",
    "check_data_layout",
    # datasets
    "LabeledImageDataset", "UnlabeledImageDataset", "MultiViewImageDataset",
    # transforms
    "transforms",
    "supervised_train", "supervised_eval", "train_tf", "eval_tf", "eval_tf_full",
    "simclr_aug", "simclr_aug_no_crop",
    "fixmatch_weak", "fixmatch_strong",
    # prompts
    "DEFAULT_TEMPLATES",
    "SCIENTIFIC_NAMES",
    "SYNONYMS",
    "build_class_prompts",
    "humanize",
    "load_class_names",
    # pseudo
    "PseudoLabelDataset",
]