"""
Python函数装饰器是一种高级的功能，它允许你修改或增强其他函数或方法的行为，而无需修改它们的源代码。装饰器通常用于在函数执行前后执行额外的代码，或者修改函数的输入和输出。这种技术在 Python 中非常强大，常常用于日志记录、性能分析、权限检查、缓存等方面。

装饰器本质上是一个函数，它接受一个函数作为参数，并返回一个新的函数。这个新函数可以包装（decorate）原始函数，从而改变其行为。装饰器使用 @ 符号来应用于函数，

"""

import time
from typing import Any


# 函数类型的装饰器
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds to run.")
        return result
    return wrapper


@timeit
def cal(n):
    for i in range(n):
        time.sleep(1)

cal(2)

# 类类型的装饰器，优点是可以用__init__方法来初始化状态或配置
class MyDecorator:
    def __init__(self, arg1, arg2) -> None:
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, func) -> Any:
        # __call__ 方法的主要作用是定义了类的实例在被调用时应该执行的操作。
        # 这样在调用类的实例时就像调用函数一样
        def wrapper(*args, **kwargs):
            # 封装函数，调用原始函数时会执行的额外操作，也就是你封装的目的，比如计时等。
            print(f"Decorator arguments: {self.arg1}, {self.arg2}")
            result = func(*args, **kwargs)
            return result
        return wrapper

@MyDecorator('hello', 'world')
def my_function():
    print('Inside my_function.')

my_function()