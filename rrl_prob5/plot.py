import matplotlib.pyplot as plt

# wth values you tested
# wth_values = [
#     10,20,30,40,50,60,70,80,90,100,
#     110,120,130,140,150,160,170,180,190,200
# ]

# wv_values= [0.005,0.01, 0.015, 0.02, 0.03, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# angle_error = [3.61, 3.50, 3.38, 3.37, 3.34, 3.15, 3.17, 3.04, 3.06, 3.20, 3.41, 3.67, 3.96, 4.24, 4.60, 4.91, 5.10]

# Mean absolute angle errors (degrees) from your console output
# angle_error = [
#     5.50,4.28,3.70,3.24,3.00,2.76,2.68,2.55,2.45,2.36,
#     2.38,2.34,2.38,2.48,2.34,2.53,2.26,2.68,2.19,2.41
# ]
wthd_values = [1,2,3,4,5,6,7,8]
angle_error_wthd = [2.95,2.69,2.43,2.70,2.85,3.09,3.56,5.06]

plt.figure(figsize=(7,5))
plt.plot(wthd_values, angle_error_wthd, marker='o', linestyle='--', color='crimson')

plt.xlabel(r"Pole Angular Velocity Weight $w_{\dot{\theta}}$", fontsize=12)
plt.ylabel("Mean Absolute Tracking Error (degrees)", fontsize=12)
plt.title(r"Angle Tracking Performance vs $w_{\dot{\theta}}$", fontsize=13)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("wthd_tracking_error.pdf", format='pdf', bbox_inches='tight')
print("Saved to wthd_tracking_error.pdf")

plt.show()