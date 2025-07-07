# Pulsar2 User Manual

[Web Review](https://npu.pages-git-ext.axera-tech.com/pulsar2-docs/)

## 1. Project Background

Next-generation AI toolchain *Pulsar2* User manual Public maintenance project

- Provide a unified internal display address for AI tool chain documents
- Reduce the maintenance cost of AI tool chain developers
- Reduce the learning cost of AI tool chain users

## 2. Local operation guide

### 2.1 git clone

```bash
# 待补充 git clone https://git-ext.axera-tech.com/npu/pulsar2-docs.git
```

The directory tree is as follows:

```bash
.
├── LICENSE
├── Makefile
├── README.md
├── build
│   ├── doctrees
│   └── html
├── requirements.txt
└── source                      # Document Main
    ├── appendix
    ├── conf.py
    ├── doc_update_info
    ├── examples                # Some examples are saved in .zip format. Due to the limitation of git pages, the online documentation does not support click-to-download operation
    ├── faq
    ├── index.rst
    ├── media
    ├── pulsar2
    ├── user_guides_advanced
    ├── user_guides_config
    ├── user_guides_quick
    └── user_guides_runtime
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
$ python -m http.server 8005      # 端口可以自定义
```

Then connect to the server via `ssh`,

```bash
ssh -L 8005:localhost:8005 username@server
```

Then access the local browser: `localhost:8005/index.html`

## 3. reference

- This project is based on Sphinx. For more information about Sphinx, please visit https://www.sphinx-doc.org/en/master/

## 4. Release


