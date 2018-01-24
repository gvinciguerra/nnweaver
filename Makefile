.PHONY: test docs clean

test:
	PYTHONPATH=$(PWD)/nnweaver:$(PYTHONPATH) pytest test

docs:
	sphinx-apidoc -Mf -o docs nnweaver
	cd docs && make html

clean:
	cd docs && make clean