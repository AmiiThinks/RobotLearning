### Code Documentation Guide

We use Sphinx and ReadtheDocs to document our code. Eventually it will be unified with our GitHub documentation. 

Use Google's code documentation style.


```bash
pip install sphinx
mkdir docs
cd docs
sphinx-quickstart
mkdir source/pyrst
```

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../../ROSBase/src/horde/scripts'))
```

```bash
make clean
rm source/pyrst/*

sphinx-apidoc -d 1 -f -o source/pyrst ../../ROSBase/src/horde/scripts ../../ROSBase/src/horde/scripts/tiles3.py ../../ROSBase/src/horde/scripts/test.py ../../ROSBase/src/horde/scripts/CTiles

make html
```

sphinx-apidoc -d 1 -f -o docs/sphinx/source/pyrst ROSBase/src/horde/scripts ROSBase/src/horde/scripts/tiles3.py ROSBase/src/horde/scripts/test.py ROSBase/src/horde/scripts/CTiles && cd docs/sphinx/ && make html && cd ../..
