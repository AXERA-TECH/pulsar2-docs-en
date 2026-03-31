# Pulsar2 User Manual

[Web Review](https://pulsar2-docs.readthedocs.io/en/latest/)

## 1. Project Background

Next-generation AI toolchain *Pulsar2* User manual Public maintenance project

- Provide a unified internal display address for AI tool chain documents
- Reduce the maintenance cost of AI tool chain developers
- Reduce the learning cost of AI tool chain users

## 2. Local operation guide

### 2.1 git clone

```bash
git clone https://github.com/AXERA-TECH/pulsar2-docs-en.git
```

The directory tree is as follows:

```bash
.
├── build
│   ├── doctrees
│   └── html
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
└── source
    ├── appendix
    ├── conf.py
    ├── doc_update_info
    ├── index.rst
    ├── media
    ├── other_tools
    ├── pulsar2
    ├── tool_classification
    ├── user_guides_advanced
    ├── user_guides_config
    └── user_guides_quick
```

### 2.2 Compile

Install Dependencies

```bash
pip install -r requirements.txt
```

Execute the following command in the project root directory

```bash
$ make clean
$ make html
```

### 2.3 Preview

After the compilation is complete, use the browser to view `build/html/index.html`. If you develop on a server, you can access the compiled document through `ssh` port forwarding, as follows:

First, you can use `python` to start an `http` service in the compiled `build/html/` folder,

```bash
$ cd build/html/
$ python -m http.server 8005
```

Then connect to the server via `ssh`,

```bash
ssh -L 8005:localhost:8005 username@server
```

Then access the local browser: `localhost:8005/index.html`

## 3. reference

- This project is based on Sphinx. For more information about Sphinx, please visit https://www.sphinx-doc.org/en/master/

## 4. Release


