#!/usr/bin/python3

import os


def getDocsIgnoreArr(docsIgnoreFile=".docsignore"):
    try:
        with open(docsIgnoreFile) as fp:
            lines = fp.readlines()
            filesToIgnore = [line.strip() for line in lines]
    except FileNotFoundError:
        filesToIgnore = []
    except Exception as e:
        print("Error reading .docignore file", e)

    return set(filesToIgnore)


def getAllPyFiles():
    pyFiles = set()
    dirname = "."

    for filename in os.listdir(dirname):
        if os.path.isfile(filename) and filename.endswith('.py'):
            pyFiles.add(filename)
    return pyFiles


filesToIgnore = getDocsIgnoreArr()
pyFiles = getAllPyFiles()
docsDir = "docs/"
filesToDocument = pyFiles.difference(filesToIgnore)

os.system(f"rm -rf {docsDir}")
for filename in filesToDocument:
    # print(filename)
    os.system(f"pdoc --html {filename} --force -o {docsDir}")
