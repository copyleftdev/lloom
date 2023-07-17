PROJECT = lloom
PYTHON_VERSION = 3.11
PACKAGE_VERSION = $(shell python -c "import ${PROJECT}; print(${PROJECT}.__version__)")

PYTEST_ARGS = -x -p no:warnings
PYTEST_COVERAGE = --cov-report term-missing --cov=${PROJECT}
PYTEST_DEBUG = -s
PYTEST_FOCUS = -k focus

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .chroma
	rm -rf build dist *.egg-info
	rm -rf ./test_db

cmd: clean
	python -m mycmd

lint:
	black ${PROJECT} tests
	isort ${PROJECT} tests
	ruff check ${PROJECT} tests

test: clean lint
	pytest ${PYTEST_COVERAGE} ${PYTEST_ARGS}

debug: clean
	pytest ${PYTEST_DEBUG} ${PYTEST_ARGS}

focus:
	pytest ${PYTEST_DEBUG} ${PYTEST_ARGS} ${PYTEST_FOCUS}

wheel: test
	python setup.py sdist bdist_wheel