import pandas as pd

df = pd.read_csv('grid_agent_node_predictions.csv')
df_thin = df.iloc[::10]  # keep every 10th row
df_thin.to_csv('predictions_thinned.csv', index=False)