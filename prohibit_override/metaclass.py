import inspect
from typing import Any, Callable, Dict, List, Tuple


AnyCallable = Callable[..., Any]


def final(funcobj: AnyCallable) -> AnyCallable:
    setattr(funcobj, '__isfinalmethod__', True)
    return funcobj


def get_func_type(cls: type, func_name: str) -> str:
    for ancestor in cls.mro():
        func = getattr(ancestor, func_name, None)
        if not func:
            continue

        if isinstance(func, classmethod):
            return 'class method'
        elif isinstance(func, staticmethod):
            return 'static method'
        else:
            return 'member function'

    raise ValueError


class FinalMeta(type):

    def __new__(mcs, name: str, bases: Tuple[Any, ...], attrs: Dict[str, Any]) -> Any:
        cls = super().__new__(mcs, name, bases, attrs)
        for func_name, func in mcs.get_methods(cls):
            for ancestor in cls.mro():
                if ancestor in [cls, object] or not hasattr(cls, func_name):
                    continue
                ancestor_func = getattr(ancestor, func_name, None)
                if not ancestor_func or not getattr(ancestor_func, '__isfinalmethod__', False) or \
                   type(func) == type(ancestor_func) and \
                   getattr(func, '__func__', func) == getattr(ancestor_func, '__func__', ancestor_func):
                    continue

                func_type = get_func_type(ancestor, func_name)
                raise TypeError(f'Fail to declare class {name}, for override final {func_type}: {func_name}')
        return cls

    @staticmethod
    def get_methods(cls: type) -> List[Tuple[str, AnyCallable]]:
        return inspect.getmembers(cls, inspect.isfunction) + inspect.getmembers(cls, inspect.ismethod)


class A(metaclass=FinalMeta):

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
