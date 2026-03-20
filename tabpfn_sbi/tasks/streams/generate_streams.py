import argparse
import multiprocessing
import pickle
from pathlib import Path

import torch

from tabpfn_sbi.tasks.streams.simulator import GD1StreamSimulator
from tabpfn_sbi.tasks.util import allocate_prior_stream_age as PriorAge


DEFAULT_STREAMS_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "files" / "streams"
)


def simulate_gd1_stream(age):
    simulator = GD1StreamSimulator()
    return simulator(torch.tensor([[age]], dtype=torch.float32))[0]


def sample_stream_ages(num_streams):
    prior_age = PriorAge()
    return prior_age.sample((num_streams,)).view(-1, 1)


def generate_streams(num_streams, num_workers):
    ages = sample_stream_ages(num_streams)
    with multiprocessing.Pool(processes=num_workers) as pool:
        streams = pool.map(simulate_gd1_stream, [age.item() for age in ages])
    return streams, ages


def save_streams(streams, ages, output_dir, start_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, (stream, age) in enumerate(zip(streams, ages)):
        file_index = start_index + offset
        file_path = output_dir / f"gd1_stream_{file_index}.pkl"
        with open(file_path, "wb") as file_handle:
            pickle.dump({"stream": stream, "age": age.item()}, file_handle)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Simulate and save GD1 streams.")
    parser.add_argument(
        "--start", type=int, default=0, help="Starting index for output files"
    )
    parser.add_argument(
        "--num-streams", type=int, default=4, help="Number of streams to simulate"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of multiprocessing workers to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STREAMS_OUTPUT_DIR,
        help="Directory where generated stream pickles are stored",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    num_workers = args.num_workers or multiprocessing.cpu_count()
    streams, ages = generate_streams(args.num_streams, num_workers)
    save_streams(streams, ages, args.output_dir, args.start)

    print(f"Output directory: {args.output_dir}")
    print(f"Starting index: {args.start}")
    print(f"Saved {len(streams)} streams to {args.output_dir}")


if __name__ == "__main__":
    main()
