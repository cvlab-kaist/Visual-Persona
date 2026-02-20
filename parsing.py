import argparse
from schp.parse_body import load_parser_model, parse
from pathlib import Path

SCHP_CHECKPOINT = Path("./pretrained_models/exp-schp-201908301523-atr.pth")


def parsing(parser_model, input_image_path, labels):
    input_image_path = str(input_image_path)  # Convert Path to string

    parsed_masks = parse(parser_model, input_image_path)

    if input_image_path is None:
        return []

    masks_with_labels = [(mask, label) for mask, label in zip(parsed_masks, labels)]
    return masks_with_labels


def save_masks(masks_with_labels, output_dir, img_id, mapping):
    output_dir = Path(output_dir)  # Ensure output_dir is a Path object

    (output_dir / "full_body_masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "body_part_masks" / img_id).mkdir(parents=True, exist_ok=True)

    for mask, label in masks_with_labels:
        if mapping[label] == 0:
            mask.save(output_dir / f"full_body_masks/{img_id}.png")
        else:
            mask.save(output_dir / f"body_part_masks/{img_id}/{mapping[label]}.png")


def main(input_image_path, output_dir):
    mapping = {
        "full body": 0,
        "head": 1,
        "torso": 2,
        "bottom": 3,
        "shoes": 4
    }

    if not SCHP_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SCHP checkpoint not found at '{SCHP_CHECKPOINT}'. "
            "Please download it from: "
            "https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP/view"
        )

    parser_model = load_parser_model(str(SCHP_CHECKPOINT))
    labels = ["full body", "head", "torso", "bottom", "shoes"]

    input_image_path = Path(input_image_path)  # Ensure it is a Path object
    img_id = input_image_path.stem  # Get filename without extension

    masks_with_labels = parsing(parser_model, input_image_path, labels)
    save_masks(masks_with_labels, output_dir, img_id, mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse human images into body region masks using SCHP.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a single image file or a directory of images.")
    parser.add_argument("--output_dir", type=str, default="./parsing",
                        help="Directory to save parsed masks (default: ./parsing).")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if input_path.is_dir():
        images = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))
    else:
        images = [input_path]

    for img in images:
        main(img, output_dir)