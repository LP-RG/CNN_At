import argparse, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from random_multiplier_generator import generate_random_multipliers, multiplier_folder
from res_net_training import new_training_method


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n-multipliers",
        type=int,
        default=1000,
        help="Number of multipliers to generate (default: 1'000)",
    )
    parser.add_argument(
        "-m",
        "--max-workers",
        type=int,
        default=5,
        help="Number of threads to run at the same time (default: 5)",
    )
    parser.add_argument(
        "-b",
        "--bit-width",
        type=int,
        default=4,
        help="Bitwidth of single multiplier input (default: 4)",
    )

    args = parser.parse_args()

    print("Generating random mulitpliers...", end=" ", flush=True)
    # generate_random_multipliers(args.bit_width, args.n_multipliers)
    print("Done")

    multiplier_files = [
        os.path.join(multiplier_folder, f)
        for f in os.listdir(multiplier_folder)
        if f.endswith('.npy')
    ]

    def process_file(file_path):
        new_training_method(
            file_path,
            pretrained=True,
            retrain=True,
            conv_type=3,
            bit_width=args.bit_width,
            signed=False,
            epochs=3,
        )

    print(f"Launching threadpool with {args.max_workers} threads")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in multiplier_files}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass
