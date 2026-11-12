# ============================================================
# MIT-BIH ARRHYTHMIA DATABASE
# COMPLETE 48-RECORD ECG FEATURE EXTRACTION
#
# One row = one beat from one ECG channel
#
# Output:
# master_MITBIH_48_records_features.csv
#
# ============================================================

import os
import glob
import warnings

import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT FOLDER
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

print("=" * 80)
print("PROJECT FOLDER")
print("=" * 80)
print(PROJECT_DIR)


# ============================================================
# 2. FIND ALL .HEA FILES
# ============================================================

hea_files = glob.glob(
    os.path.join(
        PROJECT_DIR,
        "**",
        "*.hea"
    ),
    recursive=True
)


# Remove transformed x_mitdb records
hea_files = [
    file
    for file in hea_files
    if not os.path.basename(file).lower().startswith("x_")
]


# ============================================================
# 3. BUILD RECORD PATHS
# ============================================================

record_paths = {}

for hea_file in hea_files:

    record_id = os.path.splitext(
        os.path.basename(hea_file)
    )[0]

    record_folder = os.path.dirname(
        hea_file
    )

    record_paths[record_id] = os.path.join(
        record_folder,
        record_id
    )


record_ids = sorted(
    record_paths.keys()
)


# ============================================================
# 4. DISPLAY RECORD INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("RECORDS FOUND")
print("=" * 80)

print(
    "Number of records:",
    len(record_ids)
)

print(
    record_ids
)


if len(record_ids) == 0:

    print(
        "\nERROR: No MIT-BIH records were found."
    )

    print(
        "Make sure the MIT-BIH database folder is inside:"
    )

    print(PROJECT_DIR)

    raise SystemExit


# ============================================================
# 5. EXPECTED NUMBER OF RECORDS
# ============================================================

if len(record_ids) != 48:

    print("\nWARNING:")
    print(
        "Expected 48 records, but found:",
        len(record_ids)
    )

    print(
        "The program will process the records that were found."
    )


# ============================================================
# 6. MIT-BIH BEAT ANNOTATION SYMBOLS
# ============================================================

beat_symbols = {
    "N",
    "L",
    "R",
    "B",
    "A",
    "a",
    "J",
    "S",
    "V",
    "r",
    "F",
    "e",
    "j",
    "n",
    "E",
    "/",
    "f",
    "Q"
}


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def get_sample(waves, wave_name, beat_index):

    try:

        value = waves[wave_name][beat_index]

        if value is None:
            return np.nan

        if not np.isfinite(value):
            return np.nan

        return int(value)

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError
    ):

        return np.nan


def sample_to_seconds(sample, fs):

    if pd.isna(sample):
        return np.nan

    return float(sample) / fs


def duration_ms(start, end, fs):

    if pd.isna(start) or pd.isna(end):
        return np.nan

    if end < start:
        return np.nan

    return (
        (end - start)
        / fs
        * 1000.0
    )


def amplitude(signal, sample):

    if pd.isna(sample):
        return np.nan

    sample = int(sample)

    if sample < 0 or sample >= len(signal):
        return np.nan

    return float(signal[sample])


# ============================================================
# 8. STORAGE
# ============================================================

all_rows = []

successful_records = []

failed_records = []

total_channels = 0

successful_channels = 0

failed_channels = 0


# ============================================================
# 9. PROCESS ALL RECORDS
# ============================================================

for record_number, record_id in enumerate(
    record_ids,
    start=1
):

    print("\n")
    print("=" * 80)

    print(
        f"PROCESSING RECORD {record_number}/{len(record_ids)} : {record_id}"
    )

    print("=" * 80)


    try:

        record_path = record_paths[record_id]


        # ====================================================
        # LOAD RECORD
        # ====================================================

        record = wfdb.rdrecord(
            record_path
        )

        fs = float(
            record.fs
        )

        signal_length = record.sig_len

        channel_names = record.sig_name

        number_of_channels = len(
            channel_names
        )


        print(
            "Sampling frequency:",
            fs,
            "Hz"
        )

        print(
            "Channels:",
            channel_names
        )

        print(
            "Number of samples:",
            signal_length
        )


        # ====================================================
        # LOAD ANNOTATIONS
        # ====================================================

        ann = wfdb.rdann(
            record_path,
            "atr"
        )


        # ====================================================
        # KEEP ONLY BEAT ANNOTATIONS
        # ====================================================

        beat_data = []

        for sample, symbol in zip(
            ann.sample,
            ann.symbol
        ):

            if symbol in beat_symbols:

                beat_data.append(
                    (
                        int(sample),
                        symbol
                    )
                )


        if len(beat_data) == 0:

            print(
                "No beat annotations found."
            )

            failed_records.append(
                (
                    record_id,
                    "No beat annotations"
                )
            )

            continue


        # ====================================================
        # BEAT LOCATIONS
        # ====================================================

        rpeaks = np.array(
            [
                sample
                for sample, symbol
                in beat_data
            ],
            dtype=int
        )


        beat_labels = [
            symbol
            for sample, symbol
            in beat_data
        ]


        print(
            "Annotated beats:",
            len(rpeaks)
        )


        record_channel_success = False


        # ====================================================
        # PROCESS EACH CHANNEL
        # ====================================================

        for channel_index, channel_name in enumerate(
            channel_names
        ):

            total_channels += 1

            print(
                "\n  Processing channel:",
                channel_name
            )


            # ------------------------------------------------
            # GET SIGNAL
            # ------------------------------------------------

            ecg = record.p_signal[
                :,
                channel_index
            ]


            # ------------------------------------------------
            # ECG DELINEATION
            # ------------------------------------------------

            try:

                signals, waves = nk.ecg_delineate(
                    ecg,
                    rpeaks,
                    sampling_rate=fs,
                    method="dwt",
                    show=False
                )

            except Exception as e:

                print(
                    "  Delineation failed:"
                )

                print(
                    " ",
                    str(e)
                )

                failed_channels += 1

                continue


            successful_channels += 1

            record_channel_success = True


            print(
                "  Delineation successful"
            )


            # =================================================
            # EXTRACT FEATURES FOR EACH BEAT
            # =================================================

            for i in range(
                len(rpeaks)
            ):

                r = int(
                    rpeaks[i]
                )


                # =============================================
                # P WAVE
                # =============================================

                p_on = get_sample(
                    waves,
                    "ECG_P_Onsets",
                    i
                )

                p_peak = get_sample(
                    waves,
                    "ECG_P_Peaks",
                    i
                )

                p_off = get_sample(
                    waves,
                    "ECG_P_Offsets",
                    i
                )


                # =============================================
                # QRS
                # =============================================

                qrs_on = get_sample(
                    waves,
                    "ECG_R_Onsets",
                    i
                )

                qrs_off = get_sample(
                    waves,
                    "ECG_R_Offsets",
                    i
                )


                # =============================================
                # T WAVE
                # =============================================

                t_on = get_sample(
                    waves,
                    "ECG_T_Onsets",
                    i
                )

                t_peak = get_sample(
                    waves,
                    "ECG_T_Peaks",
                    i
                )

                t_off = get_sample(
                    waves,
                    "ECG_T_Offsets",
                    i
                )


                # =============================================
                # RR INTERVAL
                # =============================================

                if i > 0:

                    rr_samples = (
                        rpeaks[i]
                        -
                        rpeaks[i - 1]
                    )

                    rr_ms = (
                        rr_samples
                        / fs
                        * 1000.0
                    )

                    if rr_ms > 0:

                        heart_rate = (
                            60000.0
                            / rr_ms
                        )

                    else:

                        heart_rate = np.nan

                else:

                    rr_ms = np.nan

                    heart_rate = np.nan


                # =============================================
                # CREATE FEATURE ROW
                # =============================================

                row = {

                    # -----------------------------------------
                    # IDENTIFICATION
                    # -----------------------------------------

                    "record_id":
                        record_id,

                    "channel":
                        channel_name,

                    "channel_index":
                        channel_index,

                    "beat_index":
                        i,

                    "beat_label":
                        beat_labels[i],


                    # -----------------------------------------
                    # BASIC R-PEAK INFORMATION
                    # -----------------------------------------

                    "R_sample":
                        r,

                    "R_time_s":
                        r / fs,

                    "RR_interval_ms":
                        rr_ms,

                    "Heart_rate_bpm":
                        heart_rate,


                    # -----------------------------------------
                    # R AMPLITUDE
                    # -----------------------------------------

                    "R_amplitude_mV":
                        amplitude(
                            ecg,
                            r
                        ),


                    # -----------------------------------------
                    # P WAVE
                    # -----------------------------------------

                    "P_onset_sample":
                        p_on,

                    "P_onset_s":
                        sample_to_seconds(
                            p_on,
                            fs
                        ),

                    "P_peak_sample":
                        p_peak,

                    "P_peak_s":
                        sample_to_seconds(
                            p_peak,
                            fs
                        ),

                    "P_amplitude_mV":
                        amplitude(
                            ecg,
                            p_peak
                        ),

                    "P_offset_sample":
                        p_off,

                    "P_offset_s":
                        sample_to_seconds(
                            p_off,
                            fs
                        ),

                    "P_duration_ms":
                        duration_ms(
                            p_on,
                            p_off,
                            fs
                        ),


                    # -----------------------------------------
                    # QRS
                    # -----------------------------------------

                    "QRS_onset_sample":
                        qrs_on,

                    "QRS_onset_s":
                        sample_to_seconds(
                            qrs_on,
                            fs
                        ),

                    "QRS_offset_sample":
                        qrs_off,

                    "QRS_offset_s":
                        sample_to_seconds(
                            qrs_off,
                            fs
                        ),

                    "QRS_duration_ms":
                        duration_ms(
                            qrs_on,
                            qrs_off,
                            fs
                        ),


                    # -----------------------------------------
                    # PR INTERVAL
                    # -----------------------------------------

                    "PR_interval_ms":
                        duration_ms(
                            p_on,
                            qrs_on,
                            fs
                        ),


                    # -----------------------------------------
                    # QT INTERVAL
                    # -----------------------------------------

                    "QT_interval_ms":
                        duration_ms(
                            qrs_on,
                            t_off,
                            fs
                        ),


                    # -----------------------------------------
                    # T WAVE
                    # -----------------------------------------

                    "T_onset_sample":
                        t_on,

                    "T_onset_s":
                        sample_to_seconds(
                            t_on,
                            fs
                        ),

                    "T_peak_sample":
                        t_peak,

                    "T_peak_s":
                        sample_to_seconds(
                            t_peak,
                            fs
                        ),

                    "T_amplitude_mV":
                        amplitude(
                            ecg,
                            t_peak
                        ),

                    "T_offset_sample":
                        t_off,

                    "T_offset_s":
                        sample_to_seconds(
                            t_off,
                            fs
                        ),

                    "T_duration_ms":
                        duration_ms(
                            t_on,
                            t_off,
                            fs
                        )
                }


                all_rows.append(
                    row
                )


        # ====================================================
        # RECORD STATUS
        # ====================================================

        if record_channel_success:

            successful_records.append(
                record_id
            )

            print(
                "\nRecord completed successfully:",
                record_id
            )

        else:

            failed_records.append(
                (
                    record_id,
                    "All channels failed"
                )
            )

            print(
                "\nRecord failed:",
                record_id
            )


    except Exception as e:

        print(
            "\nFAILED RECORD:",
            record_id
        )

        print(
            "Reason:",
            str(e)
        )

        failed_records.append(
            (
                record_id,
                str(e)
            )
        )


# ============================================================
# 10. CREATE MASTER DATAFRAME
# ============================================================

print("\n")
print("=" * 80)
print("CREATING MASTER DATASET")
print("=" * 80)


master_df = pd.DataFrame(
    all_rows
)


# ============================================================
# 11. CHECK DATA
# ============================================================

if master_df.empty:

    print(
        "\nERROR: No features were extracted."
    )

    print(
        "CSV file will not be created."
    )

    raise SystemExit


# ============================================================
# 12. RESET INDEX
# ============================================================

master_df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 13. DATASET SIZE
# ============================================================

print(
    "\nMaster dataset shape:"
)

print(
    master_df.shape
)

print(
    "\nTotal feature rows:",
    len(master_df)
)

print(
    "Total feature columns:",
    len(master_df.columns)
)


# ============================================================
# 14. DISPLAY COLUMN NAMES
# ============================================================

print("\n")
print("=" * 80)
print("FEATURE COLUMNS")
print("=" * 80)

for number, column in enumerate(
    master_df.columns,
    start=1
):

    print(
        number,
        ":",
        column
    )


# ============================================================
# 15. RECORD SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("RECORD SUMMARY")
print("=" * 80)


record_summary = (
    master_df
    .groupby("record_id")
    .agg(
        channels=(
            "channel",
            "nunique"
        ),

        beats=(
            "beat_index",
            "nunique"
        ),

        rows=(
            "record_id",
            "size"
        )
    )
    .reset_index()
)


print(
    record_summary.to_string(
        index=False
    )
)


# ============================================================
# 16. CHANNEL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("CHANNEL SUMMARY")
print("=" * 80)


channel_summary = (
    master_df
    .groupby("channel")
    .agg(
        records=(
            "record_id",
            "nunique"
        ),

        rows=(
            "channel",
            "size"
        )
    )
    .reset_index()
)


print(
    channel_summary.to_string(
        index=False
    )
)


# ============================================================
# 17. BEAT LABEL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("BEAT LABEL SUMMARY")
print("=" * 80)


label_summary = (
    master_df[
        "beat_label"
    ]
    .value_counts()
    .sort_index()
)


print(
    label_summary.to_string()
)


# ============================================================
# 18. MISSING VALUE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("MISSING VALUE SUMMARY")
print("=" * 80)


missing_values = (
    master_df
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


print(
    missing_values.to_string()
)


# ============================================================
# 19. SAVE FINAL CSV
# ============================================================

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "master_MITBIH_48_records_features.csv"
)


print("\n")
print("=" * 80)
print("SAVING CSV FILE")
print("=" * 80)

print(
    "Saving to:"
)

print(
    OUTPUT_FILE
)


master_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 20. VERIFY CSV WAS CREATED
# ============================================================

if os.path.exists(
    OUTPUT_FILE
):

    file_size_mb = (
        os.path.getsize(
            OUTPUT_FILE
        )
        /
        (1024 * 1024)
    )

    print(
        "\nCSV FILE CREATED SUCCESSFULLY"
    )

    print(
        "File:",
        OUTPUT_FILE
    )

    print(
        "File size:",
        round(
            file_size_mb,
            2
        ),
        "MB"
    )

else:

    print(
        "\nERROR: CSV file was not created."
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL EXTRACTION SUMMARY")
print("=" * 80)

print(
    "Records found:",
    len(record_ids)
)

print(
    "Records successfully processed:",
    len(successful_records)
)

print(
    "Records failed:",
    len(failed_records)
)

print(
    "Total channels processed:",
    total_channels
)

print(
    "Successful channels:",
    successful_channels
)

print(
    "Failed channels:",
    failed_channels
)

print(
    "Total rows:",
    len(master_df)
)

print(
    "Total columns:",
    len(master_df.columns)
)

print(
    "\nFINAL CSV:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 22. FAILED RECORDS
# ============================================================

if len(failed_records) > 0:

    print("\n")
    print("=" * 80)
    print("FAILED RECORDS")
    print("=" * 80)

    for record_id, reason in failed_records:

        print(
            f"{record_id}: {reason}"
        )

print("\n")
print("=" * 80)
print("ALL PROCESSING FINISHED")
print("=" * 80)