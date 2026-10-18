def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	final_1 = []
	for i in range(len(a)):
		x = a[i]
		if len(x)!=len(b):
			return -1
		sum_all = 0
		for j in range(len(b)):
			elm = x[j]*b[j]+sum_all
			sum_all = elm
		final_1.append(sum_all) 
	return final_1