import numpy as np


def blinking(numspots, numframes=2000, avg_on_time = 20, on_fraction=0.1, subframe_blink=4):
    p_off = 1-on_fraction
    k_off = 1/(avg_on_time * subframe_blink)
    k_on =( k_off - p_off*k_off)/p_off 
    
    print(f"p_off={p_off:.3f}, k_on={k_on:.3f}, k_off={k_off:.3f}",flush=True)
    
    blinkstate = np.random.binomial(1, on_fraction, size=numspots)
    
    for f in range(numframes):
        spot_ontimes = np.zeros((numspots),dtype=np.float32)
        for i in range(subframe_blink):
            turning_on = (1 - blinkstate) * np.random.binomial(1, k_on, size=numspots)
            remain_on = blinkstate * np.random.binomial(1, 1 - k_off, size=numspots)
            blinkstate = remain_on + turning_on
            spot_ontimes += blinkstate / subframe_blink
        yield spot_ontimes
        
