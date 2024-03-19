from typing import Self

def f():
    D()

class D:
    a: int
    b = 1
    c: int = 2
    def a():
        A()

a: int
a = 2


class A:
    c: int
    def __init__(self: Self, a):
        self.a = A()
        self.b = 4


def __init__(self):
    print('a')

def f():
    def g():
        B()
    class B:
        pass
    def h():
        print('hello!')
    g()
    


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
