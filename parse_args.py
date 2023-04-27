from sys import argv

# This class is used by tree_sim.py to help parse command line arguments

class ParsedArgs:
    def __init__(self):
        self.path = argv[0]
        self.by_index = []
        self.by_flag = dict()

        flag = None
        for arg in argv[1:]:
            if arg[0] == "-":
                if flag is not None:
                    self.by_flag[flag] = None
                flag = arg
            elif flag is not None:
                if flag in self.by_flag:
                    if type(self.by_flag) == list:
                        self.by_flag[flag].append(arg)
                    else:
                        self.by_flag[flag] = [self.by_flag[flag], arg]
                self.by_flag[flag] = arg
                flag = None
            else:
                self.by_index.append(arg)
        if flag is not None:
            self.by_flag[flag] = None

