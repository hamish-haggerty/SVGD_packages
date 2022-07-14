#Test all the basic functions defined in SVGD_classes
from Base_Stein.SVGD_classes import *
import pytest
import logging

#TODO: How (or should we) test base_SVGD ? Come back to this (if needed)
def test1_base_SVGD():

    b=base_SVGD(N=100,zdim=1)

    assert True

#TODO: We want to "run" this function once and use as input to several other functions

#Fixture to train model to use as input to other tests
@pytest.fixture(scope="function")
def train(request):

    d=request.param
    (N,zdim,eta,steps)=(d['N'],d['zdim'],d['eta'],d['steps'])

    svgd = trainable_SVGD(N=N,zdim=zdim,eta=eta)

    for l in range(steps):
        g = svgd.get_gradient() # "Like forward"
        svgd.AdamStep(gradient=g) #"Like step"

    return svgd.Particles

#Simple test of train
@pytest.mark.parametrize("train", [dict(N=100,zdim=1,eta=0.02,steps=100)], indirect=["train"])
def test1_train(train):

    assert torch.mean(train)<0.2
