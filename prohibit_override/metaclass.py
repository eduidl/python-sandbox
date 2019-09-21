import inspect


def final(funcobj):
    funcobj.__isfinalmethod__ = True
    return funcobj


def get_func_type(cls, func_name):
    for ancestor in cls.__mro__:
        func = ancestor.__dict__.get(func_name, None)
        if not func:
            continue

        if isinstance(func, classmethod):
            return 'class method'
        elif isinstance(func, staticmethod):
            return 'static method'
        else:
            return 'member function'


class FinalMeta(type):

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        for func_name, func in mcs.get_methods(cls):
            for ancestor in cls.__mro__:
                if ancestor in [cls, object] or func_name not in cls.__dict__:
                    continue
                ancestor_func = getattr(ancestor, func_name, None)
                if not ancestor_func or not getattr(ancestor_func, '__isfinalmethod__', False) or \
                   type(func) == type(ancestor_func) and \
                   getattr(func, '__func__', func) == getattr(ancestor_func, '__func__', ancestor_func):
                    continue

                func_type = get_func_type(ancestor, func_name)
                raise TypeError(f'Overriding @final {func_type}: {func_name}() on definition of class {name}')
        return cls

    @staticmethod
    def get_methods(cls):
        return inspect.getmembers(cls, inspect.isfunction) + inspect.getmembers(cls, inspect.ismethod)


class A(metaclass=FinalMeta):

    @final
    def final_member(self):
        pass

    @classmethod
    @final
    def final_class(cls):
        pass

    @staticmethod
    @final
    def final_static():
        pass

    def overridable(self):
        print("from A")


try:

    class B(A):

        def final_member(self):
            pass
except TypeError as e:
    print(e)

try:

    class C(A):
        pass

    class D(C):

        def final_member(self):
            pass
except TypeError as e:
    print(e)

try:

    class E(A):

        @classmethod
        def final_member(cls):
            pass
except TypeError as e:
    print(e)

try:

    class F(A):

        @classmethod
        def final_class(cls):
            pass
except TypeError as e:
    print(e)

try:

    class G(A):

        @staticmethod
        def final_static():
            pass
except TypeError as e:
    print(e)


class H(A):

    def overridable(self):
        print("from H")


H().overridable()
