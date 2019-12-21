

def yielder():
    i = 0
    i = i+1
    yield i


yielderrr = yielder()

print(yielderrr.__next__())