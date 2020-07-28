import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parse.add_argument("--batch_size", type=int, default=200, help="batch size")
    parse.add_argument("--new_model", default=False, help="new model", action="store_true")

    arg = parse.parse_args()

    print(arg.lr)
    print(arg.batch_size)
    print(arg.new_model)
