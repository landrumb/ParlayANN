import matplotlib.pyplot as plt

# Define file paths
faiss_log_file = "faiss_log.txt"
our_log_file = "our_log.txt"

# Function to read log files
def read_log_file(log_file):
    sizes = []
    times = []
    with open(log_file, 'r') as f:
        next(f)  # Skip the header
        for line in f:
            size, time = line.strip().split(',')
            sizes.append(int(size))
            times.append(float(time))
    return sizes, times

# Calculate throughput
def calculate_throughput(sizes, times):
    return [size / time for size, time in zip(sizes, times)]

# Read data from both logs
faiss_sizes, faiss_times = read_log_file(faiss_log_file)
our_sizes, our_times = read_log_file(our_log_file)

# Calculate throughput
faiss_throughput = calculate_throughput(faiss_sizes, faiss_times)
our_throughput = calculate_throughput(our_sizes, our_times)

# Plot throughput
plt.figure(figsize=(10, 6))
plt.plot(faiss_sizes, faiss_throughput, marker='o', label='FAISS')
plt.plot(our_sizes, our_throughput, marker='s', label='Our Implementation')

# Format x-axis ticks
plt.xticks(faiss_sizes, [f"{size // 1000}" for size in faiss_sizes], fontsize=10)

# Add labels, legend, and title
plt.xlabel('Dataset Size (thousands)', fontsize=12)
plt.ylabel('Throughput (Items per Second)', fontsize=12)
# plt.title('Throughput Comparison: FAISS vs Our Implementation', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()

# Save and show the plot
plt.savefig("../plots/throughput_faiss_comparison.png")
plt.show()