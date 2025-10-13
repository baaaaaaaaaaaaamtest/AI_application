import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))



from langchain_experimental.tools import PythonAstREPLTool

python_repl = PythonAstREPLTool()

python_code = """ 
import matplotlib.pyplot as plt\nimport numpy as np\n\n# Data for Ulsan average temperatures in Celsius for Jan, Feb, Mar\nmonths = ['January', 'February', 'March']\naverage_temperatures = [-3.5, -0.8, 4.2]\n\n# Create the bar chart\nplt.figure(figsize=(8, 6))\nbars = plt.bar(months, average_temperatures, color=['skyblue', 'lightcoral', 'lightgreen'])\n\n# Add titles and labels\nplt.title('Average Temperatures in Ulsan (January-March)')\nplt.xlabel('Month')\nplt.ylabel('Average Temperature (°C)')\nplt.ylim(min(average_temperatures) - 2, max(average_temperatures) + 2) # Adjust y-axis limits for better visualization\n\n# Add the temperature values on top of the bars\nfor bar in bars:\n    yval = bar.get_height()\n    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}°C', va='bottom' if yval > 0 else 'top', ha='center') # Adjust vertical alignment based on value\n\n# Add a horizontal line at 0°C for reference\nplt.axhline(0, color='grey', linestyle='--', linewidth=0.8)\n\n# Improve layout and save the figure\nplt.tight_layout()\n
plt.savefig('ulsan_average_temperatures.png', dpi=300)
"""
# print(python_code)
result = python_repl.run(python_code)
print(result)
