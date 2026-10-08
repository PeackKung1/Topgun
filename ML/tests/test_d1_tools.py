import numpy as np

from tools.train_d1 import PLAN, augment_one, class_weights, normalise


def test_augmentation_is_seeded_finite_and_rgb():
    image=np.full((224,224,3),(130,80,40),np.uint8)
    a=augment_one(image,np.random.default_rng(5)); b=augment_one(image,np.random.default_rng(5))
    assert np.array_equal(a,b)
    assert a.shape==image.shape and a.dtype==np.uint8
    assert a[...,0].mean()>a[...,1].mean()>a[...,2].mean()


def test_normalisation_matches_shared_runtime():
    pixels=np.zeros((2,2,224,224,3),np.uint8)
    x=normalise(pixels)
    assert x.shape==(2,2,3,224,224) and x.dtype==np.float32
    np.testing.assert_allclose(x[0,0,:,0,0],-np.array([.485,.456,.406])/np.array([.229,.224,.225]),rtol=1e-6)


def test_class_balance_and_predeclared_training_only_protocol():
    y=np.array([0,1,1,2,2,2])
    w=class_weights(y)
    assert np.allclose(w,[2,1,2/3])
    assert PLAN['training_cap_seed']==20261006
    assert PLAN['low_conf_threshold']==0
    assert len(PLAN['confirmation_seeds'])==3
