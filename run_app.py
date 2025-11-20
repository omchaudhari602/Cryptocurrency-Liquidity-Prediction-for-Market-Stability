import sys
import os
from streamlit.web import cli as stcli

# 1. Force the working directory to be where this script is
# This ensures app.py can find the .pkl files relative to itself
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

if __name__ == '__main__':
    # 2. Mimic the command line command "streamlit run app.py"
    sys.argv = ["streamlit", "run", "app.py"]
    
    # 3. Start Streamlit
    print(f" Starting Streamlit from: {current_dir}")
    sys.exit(stcli.main())