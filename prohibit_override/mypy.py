from typing_extensions import final

class Base:
    @final
    def hello(self) -> None:
        print("hello")

class Derived(Base):
    def hello(self) -> None:
        print("こんにちは")

Base().hello()
Derived().hello()
