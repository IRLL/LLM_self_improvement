import matplotlib.pyplot as plt

# Data
x_coords = list(range(1,11))
rouge_scores = [0.6065346003, 0.6080722809, 0.6471672344, 0.6125906706, 0.708,
                0.5980482578, 0.57, 0.571, 0.5582, 0.556]
rouge_scores = [i*100 for i in rouge_scores]
# Plotting the data
plt.figure(figsize=(8,6))
plt.plot(x_coords, rouge_scores, marker='o', linestyle='-',color="#2ca25f")
plt.xlabel("Number of Iterations")
plt.ylabel("ROUGE score in %")
plt.title("ROUGE Score Over Iterations")
plt.grid(True)
plt.savefig("1.png")
