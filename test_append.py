x = []
a = [1,2,3,4,5]
b = [2,1,4,5,6]
x.append(a)
x.append(b)

print(a)
print(b)
print(len(x))
print([max(x[i]) for i in range(len(x))])
