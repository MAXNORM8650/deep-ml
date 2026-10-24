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
		other = other if isinstance(other, Value) else Value(other)
		output = Value(self.data+other.data, _children=(self,other), _op='+')
		def _backward():
			# c=a+b
			self.grad+=output.grad
			other.grad+=output.grad
		output._backward = _backward
		return output


	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		output = Value(self.data*other.data, _children=(self,other), _op='*')

		def _backward():
			# c=a*b, 
			self.grad+=output.grad*other.data
			other.grad+=output.grad*self.data
		output._backward = _backward
		return output

	def relu(self):
		output = Value(0 if self.data<0 else self.data, (self,), 'ReLU')
		def _backward():
			# c=relu(a),
			self.grad+=(self.data>0)*output.grad
		output._backward = _backward
		return output

	def backward(self):
		# We create a full grad graph 
		topology = []
		visited = set()
		def visit_all(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					visit_all(child)
				topology.append(v)
		visit_all(self)
		self.grad=1
		for v in reversed(topology):
			v._backward()
		


