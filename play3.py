X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
Y = [ 0,   1,   1,   0,   1,   2,   2,   0,   1 ]

zipped=sorted(zip(Y,X), reverse=True)

Y, X = zip(*zipped)

print(Y)
print(X[:-1])


