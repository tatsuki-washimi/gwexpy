import logging

import numpy as np


def test_start_stop_sequence(main_window, qtbot):
    main_window.start_animation()
    assert main_window.timer.isActive() is True
    assert main_window.btn_start.isEnabled() is False
    assert main_window.btn_pause.isEnabled() is True
    assert main_window.btn_resume.isEnabled() is False

    main_window.pause_animation()
    assert main_window.timer.isActive() is False
    assert main_window.btn_pause.isEnabled() is False
    assert main_window.btn_resume.isEnabled() is True

    main_window.resume_animation()
    assert main_window.timer.isActive() is True
    assert main_window.btn_pause.isEnabled() is True
    assert main_window.btn_resume.isEnabled() is False

    main_window.stop_animation()
    qtbot.waitUntil(lambda: main_window.timer.isActive() is False)
    assert main_window.btn_start.isEnabled() is True
    assert main_window.nds_latest_raw is None
    assert main_window.time_counter == 0.0


def test_setting_change_triggers_update(main_window):
    main_window.start_animation()
    main_window.timer.stop()

    main_window.nds_cache.emit_next()
    main_window.update_graphs()
    x1, y1 = main_window.traces1[0]["curve"].getData()
    assert x1 is not None
    assert len(x1) > 0

    display_combo = main_window.graph_info1["units"]["display_y"]
    display_combo.setCurrentText("dB")
    main_window.update_graphs()
    x2, y2 = main_window.traces1[0]["curve"].getData()

    assert x2 is not None
    assert len(x2) == len(x1)
    assert not np.allclose(y1, y2)


def test_failure_modes_do_not_crash(qtbot, stub_source, gui_deps, caplog):
    from gwexpy.gui.ui.main_window import MainWindow

    window = MainWindow(enable_preload=False, data_backend=stub_source)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    window.input_controls["ds_combo"].setCurrentText("NDS")
    window.meas_controls["set_all_channels"](
        [
            {"name": "TEST:CHAN1", "active": True},
        ]
    )
    window.graph_info1["graph_combo"].setCurrentText("Time Series")
    window.graph_info1["traces"][0]["active"].setChecked(True)
    window.graph_info1["traces"][0]["chan_a"].setCurrentText("TEST:CHAN1")

    window.start_animation()
    window.timer.stop()

    caplog.set_level(logging.WARNING)
    for mode in ["gap", "nan", "timestamp_regression"]:
        stub_source.set_failure_mode(mode)
        stub_source.emit_next()
        window.update_graphs()

    stub_source.set_failure_mode("exception")
    stub_source.emit_next()
    window.update_graphs()

    qtbot.waitUntil(
        lambda: "Injected backend exception" in window.status_label.text()
    )
    assert any("StubDataSource" in rec.message for rec in caplog.records)
