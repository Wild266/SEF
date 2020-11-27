import sklearn.tree
import graphviz
d = 4
dt = sklearn.tree.DecisionTreeClassifier(max_depth = d)

voices = open("voice.csv", "r").readlines()
labels = [x.replace("\"", "") for x in voices[0].split(",")]
X = []
y = []
for i in range(1, int(len(voices) * 0.9)):
	data = voices[i].split(",")
	gender = data[-1][1]
	data = data[:-1]
	data = [float(d) for d in data]
	y.append([gender])
	X.append(data)
dt.fit(X, y)
test_X = []
test_y = []
for x in range(int(len(voices) * 0.9), len(voices)):
	data = voices[x].split(",")
	gender = data[-1][1]
	data = data[:-1]
	data = [float(d) for d in data]
	test_y.append([gender])
	test_X.append(data)
print("DEPTH: ", d)
print("DIFFERENCE TREE SCORE: ",dt.score(test_X, test_y))
dot = sklearn.tree.export_graphviz(dt, feature_names = labels[:-1], class_names = ["f", "m"])
graph = graphviz.Source(dot)
graph.render("Tree")
input("Done! <Press enter to exit>")