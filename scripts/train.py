import pyrallis

from src import configs, train


@pyrallis.wrap()
def main(config: configs.Config):
    train.train(config)


if __name__ == "__main__":
    main()
