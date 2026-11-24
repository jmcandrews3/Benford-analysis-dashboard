import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="Benford Analysis Dashboard",
    layout="wide"
)

st.title("Benford's Law Analysis Dashboard")

st.write(
    "Upload a dataset to perform Benford's Law "
    "analysis and identify unusual financial transactions."
)




st.header("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file, low_memory=False)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Unable to read file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.upper()

st.success("File uploaded successfully!")

st.write("Number of rows:", len(df))
st.write("Number of columns:", len(df.columns))

st.subheader("Dataset Preview")
st.dataframe(df.head(10))




# Select column for Benford analysis

st.subheader("Select Column for Analysis")

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

if not numeric_columns:
    st.warning("No numeric columns found.")
    st.stop()

selected_column = st.selectbox(
    "Select the column to analyze",
    numeric_columns
)




# Filter the dataset

st.sidebar.header("Filter Data")

filtered_df = df.copy()

# Filter 1: Minimum payment amount

min_amount = st.sidebar.number_input(
    "Minimum Payment Amount",
    value=float(df[selected_column].min())
)

# Filter 2: Maximum payment amount

max_amount = st.sidebar.number_input(
    "Maximum Payment Amount",
    value=float(df[selected_column].max())
)

filtered_df = filtered_df[
    (filtered_df[selected_column] >= min_amount) &
    (filtered_df[selected_column] <= max_amount)
]

# Filter 3: Categorical filter

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

if categorical_columns:

    filter_column = st.sidebar.selectbox(
        "Select a column to filter by",
        categorical_columns
    )

    filter_options = df[filter_column].dropna().unique()

    selected_values = st.sidebar.multiselect(
        "Select values to include",
        filter_options
    )

    if selected_values:
        filtered_df = filtered_df[
            filtered_df[filter_column].isin(selected_values)
        ]

# Display filtered dataset

st.subheader("Filtered Dataset")

st.write("Original row count:", len(df))
st.write("Filtered row count:", len(filtered_df))

st.dataframe(filtered_df.head(100))

if filtered_df.empty:
    st.warning("No transactions match your filters.")
    st.stop()




    # Benford's Law Analysis

import numpy as np

st.subheader("Benford's Law Analysis")

# Prepare filtered payment amounts

amounts = pd.to_numeric(
    filtered_df[selected_column],
    errors="coerce"
).dropna().abs()

amounts = amounts[np.isfinite(amounts)]
amounts = amounts[amounts > 0]

if amounts.empty:
    st.warning("No valid nonzero amounts to analyze.")
    st.stop()

# Extract significant digits

normalized = amounts / (
    10.0 ** np.floor(np.log10(amounts))
)

first_digits = np.floor(
    normalized
).astype(int)

first_two_digits = np.floor(
    normalized * 10
).astype(int)

# First digit Benford distribution

digits_1 = np.arange(1, 10)

expected_1 = np.log10(1 + 1 / digits_1)

actual_1 = (
    first_digits.value_counts(normalize=True)
    .reindex(digits_1, fill_value=0)
    .values
)

benford_1 = pd.DataFrame({
    "Digit": digits_1,
    "Actual": actual_1,
    "Expected": expected_1
})

# First two digits Benford distribution

digits_2 = np.arange(10, 100)

expected_2 = np.log10(1 + 1 / digits_2)

actual_2 = (
    first_two_digits.value_counts(normalize=True)
    .reindex(digits_2, fill_value=0)
    .values
)

benford_2 = pd.DataFrame({
    "Digits": digits_2,
    "Actual": actual_2,
    "Expected": expected_2
})

# Display results

st.write("Transactions analyzed:", len(amounts))

st.subheader("First Digit Distribution")
st.dataframe(benford_1)

st.subheader("First Two Digits Distribution")
st.dataframe(benford_2)




# Kolmogorov-Smirnov Statistical Analysis

st.subheader("KS Statistical Analysis")

# Calculate cumulative distributions

actual_cdf_1 = np.cumsum(actual_1)
expected_cdf_1 = np.cumsum(expected_1)

actual_cdf_2 = np.cumsum(actual_2)
expected_cdf_2 = np.cumsum(expected_2)

# Calculate KS statistics

ks_1 = np.max(
    np.abs(actual_cdf_1 - expected_cdf_1)
)

ks_2 = np.max(
    np.abs(actual_cdf_2 - expected_cdf_2)
)

# Calculate approximate critical value at 5%

n = len(amounts)

critical_value = 1.36 / np.sqrt(n)

# Display first digit statistical results

st.markdown("### First Digit KS Test")

st.write("KS Statistic:", round(ks_1, 6))
st.write("Critical Value:", round(critical_value, 6))

if ks_1 > critical_value:
    st.warning(
        "The actual first digit distribution is "
        "statistically different from Benford's Law."
    )
else:
    st.success(
        "The actual first digit distribution is "
        "not significantly different from Benford's Law."
    )

# Display first two digits statistical results

st.markdown("### First Two Digits KS Test")

st.write("KS Statistic:", round(ks_2, 6))
st.write("Critical Value:", round(critical_value, 6))

if ks_2 > critical_value:
    st.warning(
        "The actual first two digits distribution is "
        "statistically different from Benford's Law."
    )
else:
    st.success(
        "The actual first two digits distribution is "
        "not significantly different from Benford's Law."
    )




    # Visualize Benford Analysis Results

import matplotlib.pyplot as plt

st.subheader("Benford Analysis Visualizations")

# First digit visualization

st.markdown("### First Digit Comparison")

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.bar(
    digits_1 - 0.2,
    actual_1 * 100,
    width=0.4,
    label="Actual",
    color="steelblue"
)

ax1.bar(
    digits_1 + 0.2,
    expected_1 * 100,
    width=0.4,
    label="Expected",
    color="orange"
)

ax1.set_title("First Digit Benford Analysis")
ax1.set_xlabel("First Digit")
ax1.set_ylabel("Frequency (%)")
ax1.set_xticks(digits_1)
ax1.legend()

fig1.tight_layout()

st.pyplot(fig1)

# First two digits visualization

st.markdown("### First Two Digits Comparison")

fig2, ax2 = plt.subplots(figsize=(16, 6))

ax2.bar(
    digits_2 - 0.2,
    actual_2 * 100,
    width=0.4,
    label="Actual",
    color="steelblue"
)

ax2.bar(
    digits_2 + 0.2,
    expected_2 * 100,
    width=0.4,
    label="Expected",
    color="orange"
)

ax2.set_title("First Two Digits Benford Analysis")
ax2.set_xlabel("First Two Digits")
ax2.set_ylabel("Frequency (%)")
ax2.set_xticks(np.arange(10, 100, 5))
ax2.legend()

fig2.tight_layout()

st.pyplot(fig2)




# Descriptive Statistics

st.subheader("Descriptive Statistics")

summary_stats = filtered_df[selected_column].describe()

st.dataframe(
    summary_stats.to_frame("Value")
)

# Display key statistics

col1, col2, col3 = st.columns(3)

col1.metric(
    "Transaction Count",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average Amount",
    f"{filtered_df[selected_column].mean():,.2f}"
)

col3.metric(
    "Median Amount",
    f"{filtered_df[selected_column].median():,.2f}"
)
# Identify largest Benford deviations

st.subheader("Largest Benford Deviations")

benford_1["Difference"] = (
    benford_1["Actual"] - benford_1["Expected"]
)

benford_1["Absolute Difference"] = (
    benford_1["Difference"].abs()
)

benford_2["Difference"] = (
    benford_2["Actual"] - benford_2["Expected"]
)

benford_2["Absolute Difference"] = (
    benford_2["Difference"].abs()
)

# Sort by largest absolute difference

top_1 = benford_1.sort_values(
    "Absolute Difference",
    ascending=False
).head(5)

top_2 = benford_2.sort_values(
    "Absolute Difference",
    ascending=False
).head(5)

st.markdown("### Top 5 First Digit Deviations")
st.dataframe(top_1)

st.markdown("### Top 5 First Two Digit Deviations")
st.dataframe(top_2)

st.info(
    "Large deviations from Benford's Law may "
    "warrant further investigation. However, "
    "deviations alone do not prove fraud."
)




# Downloadable Results

st.subheader("Download Analysis Results")

# Download descriptive statistics

st.download_button(
    label="Download Summary Statistics (CSV)",
    data=summary_stats.to_csv(),
    file_name="summary_statistics.csv",
    mime="text/csv"
)

# Download first digit results

st.download_button(
    label="Download First Digit Results (CSV)",
    data=benford_1.to_csv(index=False),
    file_name="first_digit_results.csv",
    mime="text/csv"
)

# Download first two digit results

st.download_button(
    label="Download First Two Digit Results (CSV)",
    data=benford_2.to_csv(index=False),
    file_name="first_two_digit_results.csv",
    mime="text/csv"
)
# Downloadable Visualizations

import io

# First digit chart

buffer1 = io.BytesIO()

fig1.savefig(
    buffer1,
    format="png",
    dpi=300,
    bbox_inches="tight"
)

buffer1.seek(0)

st.download_button(
    label="Download First Digit Chart (PNG)",
    data=buffer1.getvalue(),
    file_name="first_digit_chart.png",
    mime="image/png"
)

# First two digits chart

buffer2 = io.BytesIO()

fig2.savefig(
    buffer2,
    format="png",
    dpi=300,
    bbox_inches="tight"
)

buffer2.seek(0)

st.download_button(
    label="Download First Two Digit Chart (PNG)",
    data=buffer2.getvalue(),
    file_name="first_two_digit_chart.png",
    mime="image/png"
)
# Downloadable Statistical Analysis

if ks_1 > critical_value:
    conclusion_1 = "Statistically significant deviation"
else:
    conclusion_1 = "No statistically significant deviation"

if ks_2 > critical_value:
    conclusion_2 = "Statistically significant deviation"
else:
    conclusion_2 = "No statistically significant deviation"

analysis_report = f"""
BENFORD'S LAW ANALYSIS REPORT

DATASET INFORMATION

File: {uploaded_file.name}
Column analyzed: {selected_column}
Original rows: {len(df):,}
Filtered rows: {len(filtered_df):,}
Transactions analyzed: {len(amounts):,}

FIRST DIGIT ANALYSIS

KS Statistic: {ks_1:.6f}
Critical Value: {critical_value:.6f}
Conclusion: {conclusion_1}

FIRST TWO DIGITS ANALYSIS

KS Statistic: {ks_2:.6f}
Critical Value: {critical_value:.6f}
Conclusion: {conclusion_2}

DESCRIPTIVE STATISTICS

{summary_stats.to_string()}

LARGEST FIRST DIGIT DEVIATIONS

{top_1.to_string(index=False)}

LARGEST FIRST TWO DIGIT DEVIATIONS

{top_2.to_string(index=False)}

INTERPRETATION

The actual leading digit distributions were compared
with the expected Benford distributions using the
Kolmogorov-Smirnov statistic.

An approximate significance level of 5% was used.
The standard critical value is conservative for
discrete leading digit distributions.

Statistically significant deviations may warrant
further investigation but do not establish fraud.

Filtering and the nature of the underlying data
may influence conformity to Benford's Law.
"""

st.download_button(
    label="Download Complete Analysis Report (TXT)",
    data=analysis_report,
    file_name="benford_analysis_report.txt",
    mime="text/plain"
)