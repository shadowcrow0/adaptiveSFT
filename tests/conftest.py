import os
import sys

# 讓 `pytest` 從 repo 根目錄或任何地方跑都找得到 adaptivesft/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
