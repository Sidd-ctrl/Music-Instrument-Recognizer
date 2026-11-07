import numpy as np

def test_cosine_distance():
    from utils import cosine_distance
    a=np.array([1,0,0]); b=np.array([0,1,0])
    assert abs(cosine_distance(a,b)-1.0) < 1e-6
    assert abs(cosine_distance(a,a)-0.0) < 1e-6

def test_extract_26d_shape():
    # synthetic sine at 440Hz
    sr=22050; dur=1.0
    t=np.linspace(0,dur,int(sr*dur), endpoint=False)
    y=np.sin(2*np.pi*440*t).astype(np.float32)
    # monkey-import inside test to avoid librosa requirement in CI
    try:
        from features import extract_26d
        vec, meta = extract_26d(y, sr)
        assert vec.shape==(26,)
        assert "centroid" in meta
    except ImportError:
        pass
