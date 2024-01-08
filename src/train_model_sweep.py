import argparse

import wandb


def train(learning_rate, batch_size):
    # Your training logic here
    print(f"Training with learning_rate={learning_rate} and batch_size={batch_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Initialize W&B run
    wandb.init(project="mlops-s222937", config=args)
    # Call the train function with hyperparameters
    train(args.learning_rate, args.batch_size)
