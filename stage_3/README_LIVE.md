Quick run:
1) Put files under repo\stage_3\
2) Edit config\providers.yaml (cities/dates).
3) Install:  powershell -ExecutionPolicy Bypass -File .\setup_stage3_live.ps1
4) Set OpenAQ token for this shell:  $env:OPENAQ_API_KEY = "<TOKEN>"
5) Create C:\Users\<You>\.cdsapirc with ADS key (see ChatGPT message).
6) Run:
   powershell -ExecutionPolicy Bypass -File .\etl_openaq_history.ps1
   powershell -ExecutionPolicy Bypass -File .\etl_cams_live.ps1
   powershell -ExecutionPolicy Bypass -File .\run_build_join_dataset.ps1
