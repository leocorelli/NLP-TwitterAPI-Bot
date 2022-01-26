install: 
	pip install --upgrade pip &&\
		pip install -r requirements.txt
	
format:
	black *.py
	
lint:
	pylint --disable=R,C,W0621 main.py
	
all: install lint