import os
from io import BytesIO
from pathlib import Path
import argparse

import rawpy
from PIL import Image
from rembg import remove, new_session


RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".rw2", ".orf"}


def load_input_image(input_path: Path) -> tuple[bytes, tuple[int, int]]:
    if input_path.suffix.lower() in RAW_EXTENSIONS:
        with rawpy.imread(str(input_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                half_size=False,
            )

        image = Image.fromarray(rgb).convert("RGB")
        original_size = image.size

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), original_size

    with Image.open(input_path) as image:
        image = image.convert("RGB")
        original_size = image.size

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), original_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a black/white or soft grayscale subject mask using rembg with a selectable local ONNX model."
    )
    parser.add_argument("input_image", nargs="?", help="Path to the input image")
    parser.add_argument("output_image", nargs="?", help="Path to the output mask image")
    parser.add_argument(
        "--model",
        choices=["bria-rmbg", "u2net", "u2netp", "birefnet-massive", "isnet", "sam"],
        default="u2net",
        help="Model to use for background removal",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional directory containing local ONNX model files",
    )
    parser.add_argument(
        "--soft-mask",
        action="store_true",
        help="Preserve soft grayscale edges instead of converting the mask to pure black and white",
    )
    args = parser.parse_args()

    if not args.input_image or not args.output_image:
        parser.error("both input_image and output_image are required")

    base_dir = Path(__file__).resolve().parent
    input_path = Path(args.input_image)
    output_path = Path(args.output_image)

    models_dir = args.model_dir
    if models_dir is not None:
        model_path = models_dir / f"{args.model}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        os.environ["U2NET_HOME"] = str(models_dir)
        model_source = f"local model file: {model_path}"
    else:
        model_path = None
        model_source = "rembg default model resolution"

    mask_type = "soft grayscale mask" if args.soft_mask else "binary mask"

    print("Processing started")
    print(f"  Input file   : {input_path}")
    print(f"  Output file  : {output_path}")
    print(f"  Model        : {args.model}")
    print(f"  Model source : {model_source}")
    print(f"  Mask type    : {mask_type}")

    session = new_session(args.model)
    input_data, original_size = load_input_image(input_path)
    output_data = remove(input_data, session=session)

    result_image = Image.open(BytesIO(output_data)).convert("RGBA")
    alpha = result_image.getchannel("A")

    if alpha.size != original_size:
        alpha = alpha.resize(original_size, Image.Resampling.LANCZOS)

    if args.soft_mask:
        mask = alpha.convert("L")
    else:
        mask = alpha.point(lambda p: 255 if p > 0 else 0).convert("L")

    mask.save(output_path)

    print(
        f"{mask_type.capitalize()} created successfully using model '{args.model}' "
        f"for {input_path.name} -> {output_path}"
    )


if __name__ == "__main__":
    main()