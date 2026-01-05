Package this Project using Pyinstaller.
Under root, executing following terms and you will get dist/homeworkgrader/:

pyinstaller --noconfirm --onedir --name HomeworkGrader ^
  --collect-all streamlit ^
  --collect-all PIL ^
  --hidden-import PIL.Image ^
  --hidden-import PIL.ImageEnhance ^
  --hidden-import dotenv ^
  --hidden-import dotenv.main ^
  --add-data "apps;apps" ^
  --add-data "core;core" ^
  launcher.py

Then put your .env document under dist/homeworkgrader.
