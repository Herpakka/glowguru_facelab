class MyClass:
    def __init__(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, new_value):
        self._value = new_value

# Usage example
obj = MyClass(10)
print(obj.get_value())  # Output: 10

obj.set_value(20)
print(obj.get_value())  # Output: 20