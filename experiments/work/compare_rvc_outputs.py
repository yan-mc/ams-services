import soundfile as sf, numpy as np, sys
a, sr = sf.read(sys.argv[1], dtype="float32")
b, _ = sf.read(sys.argv[2], dtype="float32")
n = min(len(a), len(b)); a = a[:n]; b = b[:n]
if a.ndim > 1: a = a.mean(axis=1)
if b.ndim > 1: b = b.mean(axis=1)
corr = float(np.corrcoef(a, b)[0, 1])
diff = a - b
print("sr=%d dur=%.2fs corr=%.5f rms_a=%.4f rms_b=%.4f rms_diff=%.4f max_abs_diff=%.4f mean_abs_diff=%.5f" % (
    sr, n/sr, corr, float(np.sqrt(np.mean(a**2))), float(np.sqrt(np.mean(b**2))),
    float(np.sqrt(np.mean(diff**2))), float(np.max(np.abs(diff))), float(np.mean(np.abs(diff)))))
