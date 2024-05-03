import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def logarithmic_trend(x, a, b):
    return a + b * np.log(x)
# Data
x_coords = list(range(1,12))
rouge_scores = [0.464, 0.6065346003, 0.6080722809, 0.6471672344, 0.6125906706, 0.708,
                0.5980482578, 0.57, 0.571, 0.5582, 0.556]
rouge_scores = [i*100 for i in rouge_scores]

rouge_scores2 = [0.464, 0.6365346003, 0.6380722809, 0.6671672344, 0.6825906706, 0.7321,
                0.69480482578, 0.652, 0.591, 0.49, 0.43]
rouge_scores2 = [i*100 for i in rouge_scores2]

rouge_sum = [(rouge_scores[i]+rouge_scores2[i])/2.0 for i in range(len(rouge_scores2))]


# params1, _ = curve_fit(logarithmic_trend, x_coords, rouge_sum)
# trend_line1 = logarithmic_trend(x_coords, params1[0], params1[1])
# z = np.polyfit(x_coords, np.log10(rouge_sum), 1)
# p = np.poly1d(z)
print(np.polyfit(x_coords, rouge_scores, 2))
p1 = np.poly1d(np.polyfit(x_coords, rouge_sum, 2))

y_trend = p1(x_coords)
# x_for_trend = np.linspace(1, len(x_coords), 100)

x_coord_actual = list(range(0,11))
# Plotting the data
plt.figure(figsize=(6,5))
plt.plot(x_coord_actual, rouge_scores, marker='o', linestyle='-',color='#a8ddb5', label='5-Major Voting')
plt.plot(x_coord_actual, rouge_scores2, marker='o', linestyle='-',color='#7bccc4', label='10-Major Voting')
plt.plot(x_coord_actual, y_trend, linestyle='dotted', color='#2b8cbe', label='Trend Line')
plt.legend()
plt.xlabel("Number of Iterations")
plt.ylabel("ROUGE score in %")
plt.title("ROUGE Score Over Iterations")
plt.grid(True)
plt.savefig("1.png")
