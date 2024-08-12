import json
import os
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


# Function to parse criterion output
def parse_criterion_output(directory):
    data = []
    print(f"seach in {directory}")
    # Walk through the directory and read each JSON file
    for root, g, files in os.walk(directory):
        benchmark_group = None 
        for file in files:
            if "base" in root:
                continue
            if "matmul" in root:
                if file == "benchmark.json":
                    with open(os.path.join(root, file)) as f:
                        benchmark_data = json.load(f)
                        group =  benchmark_data['group_id']
                        func =  benchmark_data['function_id']
                        size =  int(benchmark_data['value_str'])
                        throughput = benchmark_data['throughput']['Bytes']
                        benchmark_group = (group, size, throughput, func)
                if file == "estimates.json":
                    with open(os.path.join(root, file)) as f:
                        benchmark_data = json.load(f)
                        mean = benchmark_data['mean']['point_estimate']
                        if benchmark_group != None:
                            benchmark_group = (benchmark_group[0],benchmark_group[1],benchmark_group[2],benchmark_group[3],mean)
                            data.append(benchmark_group)

    return data


directory = './target/criterion/'
data = parse_criterion_output(directory)

# Function to generate comparison table
def generate_comparison_table(data):
    #group by group:
    groups = set(group for group, size, tp, func, mean in data)
    tables = []
    for group in groups:
     
        table = {}
        for g, size, tp, function_name, mean in data:
            if g == group:
                if size not in table:
                    table[size] = {}

                table[size][function_name] = mean

        print(f"Comparison Table {group}")
        print("Size\t", end="")
        functions = set(func for g, size, tp, func, mean in data if g == group)
        for function in functions:
            print(f"{function}\t", end="")
        print()
        
        for size, results in sorted(table.items()):
            print(f"{size}\t", end="")
            for function in functions:
                if function in results:
                    mean = results[function]
                    #print(f"{mean:.2f} ± {0:.2f}\t", end="")
                    print(f"{mean:.2f}\t", end="")
                else:
                    print("N/A\t", end="")
            print()
        tables.append((group, table))
    return tables

# Function to plot benchmark results
def plot_benchmark_results(data):
    groups = set(group for (group, size, tp, func, mean) in data)
    tables = []
    for group in groups:
    
        function_data = {}
        for (g, size, tp, function_name, mean) in [d for d in data if d[0] == group]:
            if function_name not in function_data:
                function_data[function_name] = {'sizes': [], 'means': [], 'tps': []}
            #if size in [1, 8, 32, 64, 128, 256, 512, 1024, 2048]:
            function_data[function_name]['sizes'].append(size)
            function_data[function_name]['means'].append(mean)
            function_data[function_name]['tps'].append(tp / mean)
        
        function_colors = {}
        c_counter = 0
        #colors = ["black", "red", "green", "blue", "yellow", "orange", "magenta", "cyan", "brown", "grey"]
        colors = plt.cm.tab20.colors

        plt.figure(figsize=(10, 14))
        for function_name, metrics in function_data.items():
            sizes = metrics['sizes']
            means = metrics['means']
            tps   = metrics['tps']

            dat = zip(sizes, means, tps)
            dat = sorted(dat, key=lambda x : x[0])
            sizes = [str(d[0]) for d in dat]
            means = [d[1] for d in dat]
            tps = [d[2] for d in dat]

            func = function_name.replace("_Prefetch", "")
            C = c_counter
            if func in function_colors:
                C = function_colors[func]
            else:
                function_colors[func] = C
                c_counter += 1
            #C = f"C{C}"
            C = colors[C]

            #std_devs = metrics['std_devs']
            #plt.errorbar(sizes, means, yerr=std_devs, label=function_name, capsize=5)
            if "Prefetch" in function_name:
                plt.plot(sizes, tps, "--", label=function_name, color=C)
            else: 
                plt.plot(sizes, tps, label=function_name, color=C)

        plt.xlabel('Size')
        plt.ylabel('Time (ns)')
        plt.ylabel('Throughput(Bytes)')
        plt.title(f'Benchmark Comparison {group}')
        #plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()


#run the analysis:
#comparison_table = generate_comparison_table(data)
plot_benchmark_results(data)
# Save comparison table to a file
# with open("benchmark_comparison_table.txt", "w") as f:
#     for (group, table) in comparison_table:
#         for size, results in sorted(table.items()):
#             f.write(f"{size}\t")
#             for function in table:
#                 if function in results:
#                     mean = results[function]
#                     #f.write(f"{mean:.2f} ± {0:.2f}\t")
#                     f.write(f"{mean:.2f}\t")
#                 else:
#                     f.write("N/A\t")
#             f.write("\n")    