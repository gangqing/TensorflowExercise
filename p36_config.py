import argparse

class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 2000
        self.batch_size = 200
        self.logdir = "logs/{name}".format(name = self.getName())
        self.save_path = "models/{name}/{name}".format(name = self.getName())
        self.simple_path = None
        self.new_model = False

    def getName(self):
        raise Exception("getName() is not redefined.")

    def __repr__(self):
        attrDict = self.get_attrs() # 字典
        result = [f"{key} = {attrDict[key]}" for key in attrDict]
        return "\n".join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self): # 获取所有属性
            if attr.startswith("_"): # 过滤系统自带的属性
                continue
            value = getattr(self,attr) # 获取属性的值
            if type(value) in [int, float, str, bool]:
                result[attr] = value
        return result

    def from_cmd(self):
        attrDict = self.get_attrs()
        parse = argparse.ArgumentParser()
        for key in attrDict:
            value = attrDict[key]
            if type(value) == bool:
                parse.add_argument(f"--{key}",default=value,help=f"default to {value}",action=f"store_{not value}".lower())
            else:
                v_type = str if type(value) is None else type(value)
                parse.add_argument(f"--{key}",type=v_type, default=value, help=f"default to {value}")

        args = parse.parse_args()
        for key in attrDict:
            if hasattr(args,key):
                setattr(self,key,getattr(args,key))


class MyConfig(Config):
    def __init__(self):
        super().__init__()
        self.epoches = 100

    def getName(self):
        return "my_config"

if __name__ == '__main__':
    config = MyConfig()
    print(config)
    config.from_cmd()
    print(config)