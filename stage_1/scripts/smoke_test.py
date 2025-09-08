import sys
from stage1_forecast.env import data_root, models_root, cache_root
print('[smoke] Python:', sys.version.split()[0])
print('[smoke] DATA_ROOT  :', str(data_root()))
print('[smoke] MODELS_ROOT:', str(models_root()))
print('[smoke] CACHE_ROOT :', str(cache_root()))
