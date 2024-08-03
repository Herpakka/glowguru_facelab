class Cal:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class calin:
    def __init__(self):
        self.add = Cal(a=0, b=0)
        self.sub = Cal(a=0, b=0)
        self.mul = Cal(a=0, b=0)
        self.div = Cal(a=0, b=0)

    def update(self, a, b, mode='add'):
        if mode == 'add':
            self.add = Cal(a, b)
        elif mode == 'sub':
            self.sub = Cal(a, b)
        elif mode == 'mul':
            self.mul = Cal(a, b)
        elif mode == 'div':
            self.div = Cal(a, b)
        self.result(mode)
        return mode, Cal(a, b)
    
    def result(self, mode='add'):
        if mode == 'add':
            return self.add.a + self.add.b
        elif mode == 'sub':
            return self.sub.a - self.sub.b
        elif mode == 'mul':
            return self.mul.a * self.mul.b
        elif mode == 'div':
            return self.div.a / self.div.b

def main(hundred=False,modes=['add','sub','mul','div']):
    if hundred:
        a = 100
        b = 100
    else:
        a = 10
        b = 10
    c = calin()
    for mode in modes:
        c.update(a, b, mode)   
main()