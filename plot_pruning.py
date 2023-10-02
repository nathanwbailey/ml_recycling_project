import matplotlib.pyplot as plt


acc_to_plot = []

with open("plot_pruning_20%_all_layers.txt", "r") as prune_file:
    for line in prune_file:
        line = line.rstrip()
        if 'Validation Accuracy' in line:
            s_idx = line.index('Validation Accuracy')
            acc = float(line[s_idx+21:])
            acc_to_plot.append(acc)


plt.plot(acc_to_plot, color='m')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.savefig('first_pruning_accuracy_plot_20%_all_layers.png')
            
            
