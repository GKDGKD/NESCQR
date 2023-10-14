import pandas as pd
import os

data_path = './result/2023_10_14_00_46_54/'
nescqr = pd.read_csv(os.path.join(data_path, 'NESCQR/metrics.csv'))
enbpi = pd.read_csv(os.path.join(data_path, 'EnbPI/metrics.csv'))
encqr = pd.read_csv(os.path.join(data_path, 'EnCQR/metrics.csv'))

print(nescqr)

methods = ['NESCQR', 'EnbPI', 'EnCQR']
dfs = [nescqr, enbpi, encqr]
concat_df = pd.concat(dfs, keys=methods, names=['Method'])
if 'Unnamed: 0' in concat_df.columns:
    concat_df.drop(columns=['Unnamed: 0'], inplace=True)

# breakpoint()
concat_df.reset_index(level=1, drop=True, inplace=True)  # 删除MultiIndex 里多余的列
# concat_df.to_csv('test.csv', index=['Method'])

## Cross
nescqr_cross = pd.read_csv(os.path.join(data_path, 'NESCQR/metrics_cross.csv'))
enbpi_cross = pd.read_csv(os.path.join(data_path, 'EnbPI/metrics_cross.csv'))
encqr_cross = pd.read_csv(os.path.join(data_path, 'EnCQR/metrics_cross.csv'))
dfs_cross = [nescqr_cross, enbpi_cross, encqr_cross]
concat_df_cross = pd.concat(dfs_cross, keys=methods, names=['Method'])
if 'Unnamed: 0' in concat_df_cross.columns:
    concat_df_cross.drop(columns=['Unnamed: 0'], inplace=True)

print(concat_df_cross)
concat_df_cross.reset_index(level=1, drop=True, inplace=True)
# concat_df_cross.to_csv('test_cross.csv', index=['Method'])


excel_file = os.path.join(data_path, 'summary.xlsx')
# 创建 ExcelWriter 对象
with pd.ExcelWriter(excel_file) as writer:
    # 将 DataFrame 写入不同 sheet
    concat_df.to_excel(writer, sheet_name='Metrics', index=['Method'])
    concat_df_cross.to_excel(writer, sheet_name='Cross', index=['Method'])

print('Done.')