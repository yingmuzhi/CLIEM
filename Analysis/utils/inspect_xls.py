import pandas as pd

xls_path = '/Volumes/T7/20260117_AIAnalysis/code/volume_surf_ins_src/analysis4_OnePageMultiComponent_Average.xls'
df = pd.read_excel(xls_path)
print(df.to_string())