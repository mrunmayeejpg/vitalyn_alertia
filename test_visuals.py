from utils.data_loader import load_patient
from visualization.charts import plot_vital_trends

df = load_patient("Deteriorating Patient")
fig = plot_vital_trends(df)
fig.show()

input("Press Enter to close the plot...")
