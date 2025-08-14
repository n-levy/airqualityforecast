## Working locations (Windows)
- Git repo (source of truth): G:\My Drive\sync\air quality forecast\Git_repo\stage1
- Python env: C:\aqf311\.venv
- Local data/models (not in git): C:\aqf311\data, C:\aqf311\models, C:\aqf311\.cache

### Recommended (mirror) run
robocopy "G:\My Drive\sync\air quality forecast\Git_repo\stage1" "C:\aqf311\repo\stage1" /MIR /XD .git .venv __pycache__ /XF *.pyc
C:\aqf311\.venv\Scripts\python.exe C:\aqf311\repo\stage1\scripts\smoke_test.py

### Direct-from-repo run (no mirror)
.\scripts\py.ps1 .\scripts\smoke_test.py
