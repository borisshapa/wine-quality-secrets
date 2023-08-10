import pyrallis

from src import configs, run


@pyrallis.wrap()
def main(config: configs.EvalConfig):
    run.eval(config)


if __name__ == "__main__":
    main()
