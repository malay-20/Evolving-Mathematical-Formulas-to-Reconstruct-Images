import numpy as np
import random
from formula import Variable, Function

class GeneticProgramming:
    
    def __init__(self, target_image):
        self.h, self.w = target_image.shape
        self.terminals = ['m', 'n', 'const']
        self.binary = ['add', 'sub', 'mul', 'div', 'min', 'max', 'mod']
        self.unary = ['sin', 'cos', 'abs', 'sqrt', 'pow2', 'exp']
        
        self.m_grid, self.n_grid = np.meshgrid(
            np.linspace(-1, 1, self.w), np.linspace(-1, 1, self.h)
        )

    def init_population(self, size, max_depth):
        pop = []
        for _ in range(size):
            depth = random.randint(2, max_depth)
            pop.append(self.random_tree(depth))
        return pop

    def random_tree(self, depth, method='grow'):
        if depth <= 1 or (method=='grow' and random.random() < 0.3):
            t = random.choice(self.terminals)
            val = random.uniform(-5, 5) if t == 'const' else None
            return Variable(t, val)
        else:
            if random.random() < 0.6:
                op = random.choice(self.binary)
                return Function(op, [self.random_tree(depth-1), self.random_tree(depth-1)])
            else:
                op = random.choice(self.unary)
                return Function(op, [self.random_tree(depth-1)])

    def evaluate_full_image(self, formula):
        try:
            if formula.depth() > 50: 
                return np.full_like(self.m_grid, 128.0)
            
            res = formula.evaluate(self.m_grid, self.n_grid)
            
            res = np.nan_to_num(res)
            res = np.clip(res, -1e5, 1e5)
            rmin, rmax = res.min(), res.max()
            
            if rmax > rmin:
                return 255 * (res - rmin) / (rmax - rmin)
            return np.full_like(self.m_grid, 128.0)
            
        except RecursionError:
            return np.full_like(self.m_grid, 128.0)

    def tournament(self, population, scores):
        indices = random.sample(range(len(population)), 3) 
        best_idx = max(indices, key=lambda i: scores[i])
        return population[best_idx].copy()

    def crossover(self, p1, p2):
        for _ in range(3):
            c1, c2 = p1.copy(), p2.copy()
            nodes1 = self.get_nodes(c1)
            nodes2 = self.get_nodes(c2)
            
            if len(nodes1) > 1 and len(nodes2) > 1:
                t1 = random.choice(nodes1[1:])
                t2 = random.choice(nodes2[1:])
                
                if type(t1) == type(t2):
                    if isinstance(t1, Variable):
                        t1.var_type, t2.var_type = t2.var_type, t1.var_type
                        t1.value, t2.value = t2.value, t1.value
                    else:
                        t1.func_type, t2.func_type = t2.func_type, t1.func_type
                        t1.children, t2.children = t2.children, t1.children
            
            if c1.depth() < 15 and c2.depth() < 15:
                return c1, c2
        
        return p1.copy(), p2.copy()

    def mutate(self, ind):
        for _ in range(2):
            child = ind.copy()
            nodes = self.get_nodes(child)
            if len(nodes) > 1:
                target = random.choice(nodes[1:])
                new_part = self.random_tree(random.randint(2, 4))
                
                if isinstance(target, Variable) and isinstance(new_part, Variable):
                    target.var_type, target.value = new_part.var_type, new_part.value
                elif isinstance(target, Function) and isinstance(new_part, Function):
                    target.func_type, target.children = new_part.func_type, new_part.children
            
            if child.depth() < 15:
                return child
        return ind.copy()

    def get_nodes(self, tree):
        try:
            nodes = [tree]
            if isinstance(tree, Function):
                for c in tree.children: 
                    nodes.extend(self.get_nodes(c))
            return nodes
        except:
            return [tree]