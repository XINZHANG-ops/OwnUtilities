from pickle import load, dump
import numpy as np


def read_pickle(filename):
    # the filename should mention the extension 'pk1'
    input = open(filename, 'rb')
    Whatever = load(input)
    input.close()
    return Whatever


def save_pickle(stuff, savename):
    output = open(savename, 'wb')
    dump(stuff, output, -1)
    output.close()


def saveList(myList, savename):
    np.save(savename, myList)


def loadList(filename, path):
    # the filename should mention the extension 'npy'
    tempNumpyArray = np.load(path + filename)
    return tempNumpyArray.tolist()
