class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def __add__(self, other):

		# Implement addition here
		other = other if isinstance(other, Value) else Value(other)
		output = Value(self.data+other.data, _children=(self, other), _op = '+')
		def _backward():
			self.grad+=output.grad # c= a+b, grad a = dl/da = dl/dc*dc/da 
			other.grad+=output.grad
		output._backward = _backward
		return output

	def __mul__(self, other):
		# Implement multiplication here
		other = other if isinstance(other, Value) else Value(other)
		output = Value(self.data*other.data, (self, other), '*')
		def _backward():
			#L = f(c); c = a*b; dl/da = dl/dc*dc/da = dl/dc*b
			self.grad+=output.grad*other.data
			other.grad+=output.grad*self.data
		output._backward = _backward
		return output

	def relu(self):
		# Implement ReLU here
		output = Value(0 if self.data<0 else self.data, (self,), 'ReLU')
		def _backward():
			# z = relu(a); dl/da = dl/dz*dz/da, 
			self.grad+= (output.data>0)*output.grad
		output._backward = _backward
		return output

	def backward(self):
		# Implement backward pass here
		topo = []
		visited_node = set()
		def nodes(v):
			if v not in visited_node:
				visited_node.add(v)
				for _child in v._prev:
					nodes(_child)
				topo.append(v)
		nodes(self)
		self.grad=1
		for v in reversed(topo):
			v._backward()
		


