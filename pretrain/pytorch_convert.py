import os
import argparse
from transformers.convert_electra_original_tf_checkpoint_to_pytorch import (
    convert_tf_checkpoint_to_pytorch,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tf_ckpt_path", type=str, default="pretrain_ckpt_len_128_small"
    )
    parser.add_argument(
        "--pt_discriminator_path", type=str, default="dialog-koelectra-small-discriminator"
    )
    parser.add_argument(
        "--pt_generator_path", type=str, default="dialog-koelectra-small-generator"
    )

    args = parser.parse_args()

    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path=args.tf_ckpt_path,
        config_file=os.path.join(args.pt_discriminator_path, "config.json"),
        pytorch_dump_path=os.path.join(args.pt_discriminator_path, "pytorch_model.bin"),
        discriminator_or_generator="discriminator",
    )

    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path=args.tf_ckpt_path,
        config_file=os.path.join(args.pt_generator_path, "config.json"),
        pytorch_dump_path=os.path.join(args.pt_generator_path, "pytorch_model.bin"),
        discriminator_or_generator="generator",
    )


if __name__ == "__main__":
    main()
