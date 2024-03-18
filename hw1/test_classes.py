class A:
    c: int
    def __init__(self: Self):
        self.a = A()
        self.b = 4


def __init__(self):
    print('a')


__init__('aaaa')


class B(A):
    pass


class C(B, A):
    pass


def f(a, b: B or None, c=3) -> int:
    pass

def f(a):
    pass

a: A = A()
print(a.a.b)
