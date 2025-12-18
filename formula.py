import numpy as np

class Node:
    def evaluate(self, m, n):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def size(self):
        return 1

    def depth(self):
        return 1


class Variable(Node):
    def __init__(self, type_, val=None):
        self.var_type = type_
        self.value = val

    def evaluate(self, m, n):
        if self.var_type == 'm':
            return m
        
        if self.var_type == 'n':
            return n
            
        return np.full_like(m, self.value)

    def copy(self):
        return Variable(self.var_type, self.value)

    def __str__(self):
        if self.var_type == 'const':
            return f"{self.value:.2f}"
        else:
            return self.var_type


class Function(Node):
    def __init__(self, type_, children):
        self.func_type = type_
        self.children = children

    def evaluate(self, m, n):
        vals = [c.evaluate(m, n) for c in self.children]
        v1 = vals[0]
        v2 = None
        if len(vals) > 1:
            v2 = vals[1]

        if self.func_type == 'add':
            return v1 + v2
        
        if self.func_type == 'sub':
            return v1 - v2
        
        if self.func_type == 'mul':
            return v1 * v2
        
        if self.func_type == 'div':
            return v1 / (np.abs(v2) + 0.001)
        
        if self.func_type == 'mod':
            return np.mod(v1, np.abs(v2) + 0.001)
        
        if self.func_type == 'min':
            return np.minimum(v1, v2)
        
        if self.func_type == 'max':
            return np.maximum(v1, v2)

        if self.func_type == 'sin':
            return np.sin(v1)
        
        if self.func_type == 'cos':
            return np.cos(v1)
        
        if self.func_type == 'abs':
            return np.abs(v1)
        
        if self.func_type == 'sqrt':
            return np.sqrt(np.abs(v1))
        
        if self.func_type == 'exp':
            return np.exp(np.clip(v1, -5, 5))
        
        if self.func_type == 'pow2':
            return np.square(np.clip(v1, -100, 100))
        
        if self.func_type == 'floor':
            return np.floor(v1)
        
        if self.func_type == 'ceil':
            return np.ceil(v1)
            
        if self.func_type == 'step':
            return (v1 > 0).astype(float)

        return v1

    def copy(self):
        return Function(self.func_type, [c.copy() for c in self.children])

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        return 1 + max(c.depth() for c in self.children)

    def __str__(self):
        args = ", ".join(str(c) for c in self.children)
        return f"{self.func_type}({args})"