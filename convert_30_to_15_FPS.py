# Load the file and remove every other line (i.e., keep every 2nd line starting from line 0)
input_path = "algo_input_data/datakoi_output.txt"

# Read the file
with open(input_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Keep every other line
filtered_lines = lines[::2]

# Create a new line with 28 "0.0" values, comma-separated
new_line = ",".join(["0.0"] * 28) + ", " + "\n"

# Append the new line
for i in range(10):
    filtered_lines.append(new_line)

# Save the filtered lines to a new file
output_path = "algo_input_data/datakoi_output_15FPS.txt"
with open(output_path, "w", encoding="utf-8") as file:
    file.writelines(filtered_lines)

output_path
