from __future__ import annotations


def test_excitation_targets_are_reflected_in_channel_selectors(main_window):
    panel = main_window.exc_controls["panels"][0]
    target_name = "CUSTOM:EXC"

    panel["active"].setChecked(True)
    panel["ex_chan"].setEditText(target_name)
    main_window.on_measurement_channel_changed()

    graph_combo = main_window.graph_info1["traces"][0]["chan_a"]
    target_combo = main_window.exc_controls["target_combos"][0]

    assert graph_combo.findText(target_name) != -1
    assert target_combo.findText("TEST:CHAN1") != -1
    assert target_combo.currentText() == target_name


def test_collect_data_map_builds_fallback_timebase_for_active_excitation(main_window):
    panel = main_window.exc_controls["panels"][0]
    panel["active"].setChecked(True)
    panel["ex_chan"].setEditText("Excitation")

    main_window.data_source = "NDS"
    main_window.nds_latest_raw = None
    main_window.input_controls["pcaudio"].setChecked(False)

    data_map, current_times, current_fs = main_window._collect_data_map()

    assert data_map == {}
    assert current_times is not None
    assert len(current_times) > 0
    assert current_fs == 16384
