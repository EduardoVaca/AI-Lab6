import matplotlib.pyplot


x = []
y = []
colors = []
for _ in range(100):    
    data = [float(x) for x in input().split(',')]
    x.append(data[0])
    y.append(data[1])
    colors.append('blue' if data[2] == 0 else 'red')
matplotlib.pyplot.scatter(x,y,color=colors)
matplotlib.pyplot.show()