import inspect
from typing import Any, Callable, List, Tuple

AnyCallable = Callable[..., Any]


def final(funcobj: AnyCallable) -> AnyCallable:
    setattr(funcobj, '__isfinalmethod__', True)
    return funcobj


def get_func_type(cls: type, func_name: str) -> str:
    func = getattr(cls, func_name)

    if isinstance(func, classmethod):
        return 'class method'
    elif isinstance(func, staticmethod):
        return 'static method'
    else:
        return 'member function'


class Final:

    def __init_subclass__(cls, **kwargs) -> None:
        for func_name, func in cls.get_methods():
            for ancestor in cls.__bases__:
                if ancestor == object or not hasattr(cls, func_name):
                    continue
                ancestor_func = getattr(ancestor, func_name, None)
                if not ancestor_func or not getattr(ancestor_func, '__isfinalmethod__', False) or \
                        type(func) == type(ancestor_func) and \
                        getattr(func, '__func__', func) == getattr(ancestor_func, '__func__', ancestor_func):
                    continue

                func_type = get_func_type(ancestor, func_name)
                raise TypeError(f'Fail to declare class {cls.__name__}, for override final {func_type}: {func_name}')

    @classmethod
    def get_methods(cls) -> List[Tuple[str, AnyCallable]]:
        return inspect.getmembers(cls, lambda x: inspect.isfunction(x) or inspect.ismethod(x))


class A(Final):

    @final
    def final_member(self) -> None:
        pass

    @classmethod
    @final
    def final_class(cls) -> None:
        pass

    @staticmethod
    @final
    def final_static() -> None:
        pass

    def overridable(self) -> None:
        print("from A")


class B(A):
    pass


try:

    class C(A):

        def final_member(self) -> None:
            pass
except TypeError as e:
    print(e)

try:

    class D(B):

        def final_member(self) -> None:
            pass
except TypeError as e:
    print(e)

try:

    class E(A, int):

        def final_member(self) -> None:
            pass
except TypeError as e:
    print(e)

try:

    class F(int, B):

        def final_member(self) -> None:
            pass
except TypeError as e:
    print(e)

try:

    class G(A):

        @classmethod
        def final_member(cls) -> None:
            pass
except TypeError as e:
    print(e)

try:

    class H(A):

        @staticmethod
        def final_member() -> None:
            pass
except TypeError as e:
    print(e)

try:

    class J(A):

        @classmethod
        def final_class(cls) -> None:
            pass
except TypeError as e:
    print(e)

try:

    class K(A):

        @staticmethod
        def final_static() -> None:
            pass
except TypeError as e:
    print(e)


class L(A):

    def overridable(self) -> None:
        print("from L")


L().overridable()
