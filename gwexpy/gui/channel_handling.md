# Channel Handling in pyaggui

This document explains how channels are handled in the `gwexpy` GUI (`pyaggui`), based on codebase analysis and implementation details.

## 1. Overview

In `gwexpy` and `pyaggui`, channels are primarily handled as **strings (channel names)**. Data structures utilize objects such as `gwpy.timeseries.TimeSeries`, with the channel name managed via the `name` attribute or as a dictionary key.

## 2. Channel Selection in the GUI

`pyaggui` follows the design philosophy of DTT (`diaggui`), separating channel selection into "Measurement Definition" and "Result Selection (Plotting)."

* **Measurement Tab**:
  * **Role**: Define the **channels to be targeted for data acquisition**.
  * Supports registering and managing many channels (e.g., up to 96).
  * Only channels marked as `Active` are targeted for data acquisition in SIM/NDS modes.

* **Result Tab (GraphPanel)**:
  * **Role**: Select the **channels to be plotted (Traces)** from those registered in the Measurement tab.
  * Each trace has input fields for `A` (Channel A) and `B` (Channel B, used for dual-channel measurements like transfer functions).
  * The combo boxes only list **channels that are marked as Active in the Measurement tab** (updated dynamically).

## 3. Processing Flow by Data Source

### A. Online (NDS) - *Verification Pending / List Retrieval Not Implemented*

1.  Enter the desired channel names in the **Measurement tab** and mark them as `Active`.
    *   Currently, manual entry is required as there is no feature to retrieve a channel list from the server.
2.  The combo boxes in the **Result tab** will display only the Active channels from the Measurement tab.
3.  Clicking "Start" triggers `start_animation`, which requests data (`NDSDataCache`) for **all Active channels** defined in the Measurement tab.

### B. File (FILE) - XML Files

Following compatibility with `diaggui`, LIGO lightweight XML (`.xml`) files behave as follows:

1.  Load the XML file via "Open...".
2.  The channel list and the **Active status** of each channel are extracted from the parameter information in the file.
3.  These states are immediately reflected (restored) in the **Measurement Master**: Entry of channel names for measurement (SIM/NDS/FILE).
* **Result Selection**: Selection of traces for plotting (A/B) from Active measurement channels.
* **Compatibility**: Restoration of DTT measurement configurations from XML files.
4.  Updating the Measurement tab automatically refreshes the combo box options in the **Results tab**.
    *   This restores the "measurement configuration" saved in the file, allowing users to select and plot traces from the restored list.

### C. File (FILE) - Others (GWF, Mini-SEED, HDF5, etc.)

**Future Work**:
Format support for `.gwf`, `.miniseed`, `.h5`, etc., is currently functional for reading (legacy behavior), but DTT-compatible features like reflecting Active states in the Measurement tab are **not yet supported**.
*   Currently, all channels in a file may appear directly as candidates in the Results tab, or the Measurement tab may be bypassed.
*   In the future, we aim to unify these formats to use the Measurement tab as the primary master configuration.

### D. Excitation (Simulation)
In **NDS (Online)** mode, the **Excitation (Simulation) tab** can be used to generate and display synthetic signals (functional even without an NDS connection).

1.  Enable a panel (`Active`) in the **Excitation (Simulation) tab** and configure waveform parameters.
2.  Target channel names (e.g., `Excitation-0`) are automatically generated and registered.
3.  The Active excitation channels are automatically added to the combo boxes in the **Result tab**.
4.  Clicking "Start" plots both the data retrieved from NDS (if any) and the locally generated waveforms.
    *   Settings in the Measurement tab are not required (Excitation channels are handled separately).

## 4. Alignment with DTT (diaggui)

Following updates on 2025/12/25, channel handling in `pyaggui` aligns with the DTT design principle where **Measurement = Master** and **Results = View**.

*   **Improvements**:
    *   Previously, channel names were entered or selected directly in the Results tab. The flow now requires (or prioritizes) definition in the Measurement tab.
    *   By restoring the `Active` flags during XML loading, users can faithfully reproduce measurement configurations saved in DTT.

## 5. Implementation Notes and Future Challenges
*   **`gwexpy/gui/ui/tabs.py`**: Holds the model (`channel_states`) for managing channel states (names, Active flags) in the `Measurement` tab and issues change signals (`measure_callback`).
*   **`gwexpy/gui/ui/main_window.py`**:
    *   `on_measurement_channel_changed`: Receives change notifications from the Measurement tab and updates the Results tab combo boxes.
    *   Uses `loaders.extract_xml_channels` during XML loading to programmatically update the Measurement tab state via `set_all_channels`.
*   **Technical Debt / Refactoring**:
    *   Previously implemented in `gwexpy/gui/io/loaders.py`, this logic has been moved and integrated into **`gwexpy/io/dttxml_common.py`**.
    *   `loaders.py` now utilizes functionalities from `dttxml_common.py` as needed.
