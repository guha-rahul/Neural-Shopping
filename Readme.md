```bash
git clone https://github.com/yourusername/neural-shopping.git

cd neural-shopping
```
# Creating the virtual env 
```bash
python3 -m venv venv
source venv/bin/activate
```
#this generates the PKL files 
```bash
pip install -r requirements.txt
python train.py
```

# Frontend 
```bash
streamlit run frontend.py
```