from collections import defaultdict
import sys

# Dictionary to store sum of values for each (nid, NIC) pair
counter_sums = defaultdict(int)  

# Read data from file
with open(f"{sys.argv[1]}", "r") as file:
    for line in file:
        parts = line.split()
        nid = parts[1]  # Node ID
        nic = parts[2]  # NIC number
        
        try:
            first_value = int(float(parts[-2]))  # Convert only the first value
            counter_sums[(nid, nic)] += first_value
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")  # Debugging output

# Print results
node_packet_counts = defaultdict(int)
for key, value in sorted(counter_sums.items()):
    print(f"{key}: {value}")
    node_packet_counts[key[0]] += value

print(node_packet_counts)

