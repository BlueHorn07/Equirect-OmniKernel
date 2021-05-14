from matplotlib import pyplot as plt
import json

f = open("log.txt", "r")

epoches = []
test_acc = []

i = 1
while True:
  line = f.readline()
  if line:
    line = line.split()

    epoches.append(i)
    test_acc.append(int(line[7][1:3]))
    i += 1
  else:
    break

acc_max = max(test_acc)
where_max = test_acc.index(acc_max) + 1

plt.title("mollweide-CNN")
plt.xlabel("epoch")
plt.ylabel("acc (%)")
plt.plot(epoches, test_acc, c='blue')
plt.axvline(x=where_max, c="red")
plt.text(x=where_max+1, y=acc_max-3, s='[%d] %d'%(where_max, acc_max))
plt.legend(['test_acc'], loc=0)
plt.show()
