---
sr-due: 2026-07-31
sr-interval: 10
sr-ease: 250
---
#code 

[original docs](https://docs.python.org/3/reference/datamodel.html#special-method-names)


## `object.__new__(_cls_[, _..._])`
用于创建类的新实例，在`__init__`之前执行

## `obj.__dict__()`
储存类的属性和方法
对于实例，只存储特别声明的属性

```python
class Foo:
    x = 1
    def bar(self):
        pass
        
obj = Foo()
obj.x = 1

Foo.__dict__
# {'__module__': '__main__', '__firstlineno__': 8, 'x': 1, 'bar': <function Foo.bar at 0x0000029F4B2516F0>, '__static_attributes__': (), '__dict__': <attribute '__dict__' of 'Foo' objects>, '__weakref__': <attribute '__weakref__' of 'Foo' objects>, '__doc__': None}

obj.__dict__
# {'x': 1}
```

