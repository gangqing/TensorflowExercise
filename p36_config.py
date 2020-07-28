import argparse


class Config:
    def __init__(self):
        self.simple_path = None
        self.save_path = "models/{name}/{name}".format(name=self.get_name())
        self.logdir = "logs/{name}".format(name=self.get_name())
        self.epoches = 200
        self.batch_size = 500
        self.lr = 0.01
        self.new_model = False

    def get_name(self):
        raise Exception("get_name() is not redefined!")

    def __repr__(self):
        """
        :return: attr1 = value1, attr2 = value2 ...
        """
        attrs = self.get_attrs()
        result = ["{attr} = {value}".format(attr=attr, value=attrs[attr]) for attr in attrs]
        return ", ".join(result)

    def get_attrs(self):
        """
        :return: dict
        """
        result = {}
        for attr in dir(self):
            if attr.startswith("_"):
                continue
            value = getattr(self, attr)  # 获取属性的值
            if type(value) in [int, float, str, bool]:
                result[attr] = value
        return result

    def from_cmd(self):
        attrs = self.get_attrs()
        parser = argparse.ArgumentParser()
        for attr in attrs:
            value = attrs[attr]
            if type(value) == bool:
                parser.add_argument("--{attr}".format(attr=attr), default=value, action=f"store_{not value}".lower())
            else:
                v_type = str if value is None else type(value)
                parser.add_argument("--{attr}".format(attr=attr), type=v_type, default=value, help=f"default to {value}")

        args = parser.parse_args()
        for attr in attrs:
            if hasattr(args, attr):
                setattr(self, attr, getattr(args, attr))


class MyConfig(Config):
    def get_name(self):
        return "36"


if __name__ == '__main__':
    config = MyConfig()
    print(config)