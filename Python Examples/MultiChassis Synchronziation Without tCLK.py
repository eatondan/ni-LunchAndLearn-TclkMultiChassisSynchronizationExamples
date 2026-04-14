import niscope
import nifgen
import nisync
import numpy as np
from nisync.constants import CLK_OUT, OSCILLATOR, CLK_IN, PXI_CLK10_IN, PXI_TRIG0, PFI0
import matplotlib.pyplot as plt

# --- Hardware Configuration Resources ---
MASTER_SCOPE = "PXI1_SCOPE1"
REST_SCOPES = ["PXI2_SCOPE2"]
MASTER_SYNC = "PXI1_MasterSync"
REST_SYNC = "PXI2_SlaveSync"
FGEN = "PXI1_FGEN1"

# --- Scope & FGEN Settings ---
CHANNEL = "0"
SAMPLE_RATE = 100e6
RECORD_LENGTH = 100_000
VERTICAL_RANGE = 2.0
VERTICAL_OFFSET = 0.0
TRIGGER_LEVEL = 0.5
REFERENCE_POSITION = 50.0

# --- Trigger & Routing Settings ---

MASTER_SCOPE_REF_TRIGGER_EXPORT = PXI_TRIG0
MASTER_SYNC_REF_TRIGGER_EXPORT = PFI0
REST_SYNC_REF_TRIGGER_IMPORT = PFI0
REST_SCOPE_REF_TRIGGER_IMPORT = PXI_TRIG0


# --- Helper Functions ---
def configure_scope(scope, is_master):
    # --- Configure the scope range ---
    print(f"Configuring: {scope.io_resource_descriptor} as Master: {is_master}")
    scope.configure_vertical(
        range=VERTICAL_RANGE,
        coupling=niscope.VerticalCoupling.DC,
        offset=VERTICAL_OFFSET,
        probe_attenuation=1.0,
        enabled=True
    )

    # --- Configure the scope timing / acquisition ---
    scope.configure_horizontal_timing(
        min_sample_rate=SAMPLE_RATE,
        min_num_pts=RECORD_LENGTH,
        ref_position=REFERENCE_POSITION,
        num_records=1,
        enforce_realtime=True
    )

    if is_master:
        # --- Configure the scope master reference trigger for analog edge and export trigger to PXI trig lines ---
        scope.configure_trigger_edge(
            trigger_source=CHANNEL,
            level=TRIGGER_LEVEL,
            trigger_coupling=niscope.TriggerCoupling.DC,
            slope=niscope.enums.TriggerSlope.POSITIVE
        )
        scope.exported_ref_trigger_output_terminal = MASTER_SCOPE_REF_TRIGGER_EXPORT

    else:
        # --- Configure the remaining scope reference trigger for digital edge coming from PXI trig lines ---
        scope.configure_trigger_digital(
            trigger_source = REST_SCOPE_REF_TRIGGER_IMPORT,
            slope=niscope.enums.TriggerSlope.POSITIVE
        )


def configure_fgen(fgen):
    # --- Configure the FGEN to output a Square wave to use for edge detection ---
    print(f"Configuring: {fgen.io_resource_descriptor}")
    fgen.output_mode = nifgen.OutputMode.FUNC   # Select "standard function" mode
    fgen.channels[CHANNEL].configure_standard_waveform(
        waveform=nifgen.Waveform.SQUARE,
        amplitude=TRIGGER_LEVEL*2,           # V pk-pk
        frequency=10_000,           # Hz
        dc_offset=0.0,               # V
        start_phase=0.0
    )              # degrees

def find_threshold_crossing(array, threshold, direction="rising"):
    # --- Determine the first threshold crossing in an array, then interpolate to find the decimal sample ---
    arr = np.asarray(array, dtype=float)
    shifted = arr - threshold
    if direction == "rising":
        mask = (shifted[:-1] < 0) & (shifted[1:] >= 0)
    elif direction == "falling":
        mask = (shifted[:-1] >= 0) & (shifted[1:] < 0)
    else:  # both
        mask = np.diff(np.signbit(shifted))
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    i = indices[0]
    fraction = (threshold - arr[i]) / (arr[i + 1] - arr[i])
    return i + fraction

def fetch_and_compare_waveforms(master_scope, rest_scope):
    # --- Build an array of fetched data from the scopes, find the decimal threshold crossing, and return values ---
    samples_array = []

    # --- Scope Fetch ---
    wfm_master = master_scope.channels[CHANNEL].fetch(num_samples=RECORD_LENGTH)[0]
    wfm_rest = rest_scope.channels[CHANNEL].fetch(num_samples=RECORD_LENGTH)[0]

    # --- Format fetched samples into array ---
    master_samples = np.asarray(wfm_master.samples, dtype=float)
    rest_samples = np.asarray(wfm_rest.samples, dtype=float)

    # --- Built single array to return ---
    samples_array.append(master_samples)
    samples_array.append(rest_samples)

    # --- Find decimal threshold crossing ---
    master_index_crossing = find_threshold_crossing(master_samples, TRIGGER_LEVEL)
    rest_index_crossing = find_threshold_crossing(rest_samples, TRIGGER_LEVEL)

    # --- Calculated sample offset and time offset ---
    sample_offset = master_index_crossing - rest_index_crossing
    time_offset = sample_offset * (1/SAMPLE_RATE)

    return samples_array, sample_offset, time_offset

# --- Main Sequence ---
with nisync.Session(MASTER_SYNC) as master_sync, nisync.Session(REST_SYNC) as rest_sync:
    # --- Connect the master sync's oscillator to clock out, connect clock in to PXI 10 in ---
    master_sync.connect_clock_terminals(OSCILLATOR, CLK_OUT)
    master_sync.connect_clock_terminals(CLK_IN, PXI_CLK10_IN)

    # --- Connect the rest sync's clock in to PXI 10 in ---
    rest_sync.connect_clock_terminals(CLK_IN, PXI_CLK10_IN)

    # --- Connect the master scope's exported reference trigger to master sync's PFI output ---
    master_sync.connect_trigger_terminals(MASTER_SCOPE_REF_TRIGGER_EXPORT, MASTER_SYNC_REF_TRIGGER_EXPORT)

    # --- Connect the rest sync's PFI input to rest scope expected reference trigger line ---
    rest_sync.connect_trigger_terminals(REST_SYNC_REF_TRIGGER_IMPORT, REST_SCOPE_REF_TRIGGER_IMPORT)

    with nifgen.Session(FGEN) as fgen:
        # --- Create fgen session ---
        with niscope.Session(MASTER_SCOPE) as master_scope, niscope.Session(REST_SCOPES[0]) as rest_scope:
            # --- Create & configure scope sessions session ---
            configure_scope(master_scope, is_master=True)
            configure_scope(rest_scope, is_master=False)

            # --- Configure fgen sessions session ---
            configure_fgen(fgen)

            # --- Start the rest of the scopes first as they wait on the refer trigger from master, then start master scope ---
            rest_scope.initiate()
            master_scope.initiate()
            print("Waiting: Scopes waiting for reference trigger")

            # --- Start square wave, should trigger master, which should trigger rest of scopes ---
            fgen.initiate()
            print(f"Generating: {fgen.io_resource_descriptor} is generating")

            # --- Fetch triggered data, calcuate offsets ---
            fetched_samples_array, calculated_sample_offset, calculated_time_offset = fetch_and_compare_waveforms(master_scope, rest_scope)

            # --- Stop generating square wave ---
            fgen.abort()

    # --- Disconnect the reference triggers ---
    master_sync.disconnect_trigger_terminals(MASTER_SCOPE_REF_TRIGGER_EXPORT, MASTER_SYNC_REF_TRIGGER_EXPORT)
    rest_sync.disconnect_trigger_terminals(REST_SYNC_REF_TRIGGER_IMPORT, REST_SCOPE_REF_TRIGGER_IMPORT)

    # --- Disconnect the clocks ---
    master_sync.disconnect_clock_terminals(OSCILLATOR, CLK_OUT)
    master_sync.disconnect_clock_terminals(CLK_IN, PXI_CLK10_IN)
    rest_sync.disconnect_clock_terminals(CLK_IN, PXI_CLK10_IN)

    # --- Print Results ---
    print(f"Result: Sample Offset: {calculated_sample_offset}")
    print(f"Result: Time Offset (sec): {calculated_time_offset:.9f} s")
    print(f"Result: Time Offset (nsec): {calculated_time_offset * 1_000_000_000:.6f} ns")

    # --- Plot Results ---
    plt.figure()
    plt.plot(fetched_samples_array[0], label=f"Master Scope")
    plt.plot(fetched_samples_array[1], label=f"Rest Scope")
    plt.title(f"Scope Trace")
    plt.legend()
    plt.show()