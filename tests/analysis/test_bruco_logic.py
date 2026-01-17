import numpy as np

from gwexpy.analysis.bruco import BrucoResult


def test_bruco_result_top_n():
    freqs = np.array([10, 20, 30])
    target_psd = np.array([1.0, 1.0, 1.0])
    res = BrucoResult(freqs, "Target", target_psd, top_n=2)

    # Batch 1: Channels A, B
    # A: [0.1, 0.9, 0.5]
    # B: [0.2, 0.8, 0.6]
    names_1 = ["A", "B"]
    coh_1 = np.array([[0.1, 0.9, 0.5], [0.2, 0.8, 0.6]])
    res.update_batch(names_1, coh_1)

    # Check
    # Bin 0 (10Hz): B(0.2), A(0.1). Top is B, A
    assert res.top_channels[0, 0] == "B"
    assert res.top_coherence[0, 0] == 0.2

    # Batch 2: Channel C
    # C: [0.95, 0.0, 0.55]
    names_2 = ["C"]
    coh_2 = np.array([[0.95, 0.0, 0.55]])
    res.update_batch(names_2, coh_2)

    # Check Bin 0: C(0.95), B(0.2). A(0.1) is pushed out
    assert res.top_channels[0, 0] == "C"
    assert res.top_channels[0, 1] == "B"
    assert res.top_coherence[0, 0] == 0.95
    assert res.top_coherence[0, 1] == 0.2

    # Check Bin 2 (30Hz): B(0.6), C(0.55). Top 2
    # But A was 0.5.
    # Order: B(0.6), C(0.55), A(0.5). Top 2 are B, C
    print(f"Bin 2 Channels: {res.top_channels[2]}")
    # Since sorts are not necessarily stable for equal values, but here distinct.
    # A=0.5, B=0.6, C=0.55 -> B, C, A. Top 2: B, C.
    assert "B" in res.top_channels[2]
    assert "C" in res.top_channels[2]
    assert "A" not in res.top_channels[2]


def test_projection():
    freqs = np.array([10])
    target_psd = np.array([4.0])
    res = BrucoResult(freqs, "Target", target_psd, top_n=1)

    res.update_batch(["A"], np.array([[0.25]]))  # sqrt(0.25) = 0.5

    proj, coh = res.get_noise_projection(0)
    # Proj = ASD * sqrt(Coh) = 2.0 * 0.5 = 1.0
    assert coh[0] == 0.25
    assert proj[0] == 1.0


if __name__ == "__main__":
    try:
        test_bruco_result_top_n()
        test_projection()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
