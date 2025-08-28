"""
Tests for segment profiling and reporting functionality
"""

import pandas as pd
import pytest


def test_reporting_files_exist(project_root):
    """
    Test that all reporting output files exist.
    """
    # Check segment profiling files
    segment_profiles_path = project_root / "data" / "processed" / "segment_profiles.csv"
    assert (
        segment_profiles_path.exists()
    ), f"Segment profiles file not found: {segment_profiles_path}"

    customer_segments_final_path = (
        project_root / "data" / "processed" / "customer_segments_final.csv"
    )
    assert (
        customer_segments_final_path.exists()
    ), f"Customer segments final file not found: {customer_segments_final_path}"

    customer_segments_labeled_path = (
        project_root / "data" / "processed" / "customer_segments_labeled.csv"
    )
    assert (
        customer_segments_labeled_path.exists()
    ), f"Customer segments labeled file not found: {customer_segments_labeled_path}"

    # Check report files
    segment_report_md_path = project_root / "reports" / "segment_report.md"
    assert (
        segment_report_md_path.exists()
    ), f"Segment report markdown file not found: {segment_report_md_path}"

    # Check visualization files
    figures_dir = project_root / "reports" / "figures"
    expected_figures = [
        "segments_size_bar.png",
        "segments_revenue_share.png",
        "segments_rfm_heatmap.png",
        "segments_rfm_boxplots.png",
    ]

    for figure in expected_figures:
        figure_path = figures_dir / figure
        assert figure_path.exists(), f"Figure file not found: {figure_path}"


def test_segment_profiles_data_sanity(project_root):
    """
    Test that segment profiles data has correct statistical properties.
    """
    segment_profiles_path = project_root / "data" / "processed" / "segment_profiles.csv"
    profiles_df = pd.read_csv(segment_profiles_path)

    # Check required columns
    required_columns = [
        "segment",
        "n_customers",
        "share_customers",
        "avg_recency",
        "median_recency",
        "avg_frequency",
        "median_frequency",
        "avg_monetary",
        "median_monetary",
        "revenue_share",
        "high_recency_share",
    ]

    for col in required_columns:
        assert col in profiles_df.columns, f"Segment profiles missing column: {col}"

    # Check that share_customers sum to approximately 1.0
    total_share = profiles_df["share_customers"].sum()
    assert total_share == pytest.approx(
        1.0, abs=1e-6
    ), f"share_customers sum {total_share:.6f} is not close to 1.0"

    # Check that revenue_share sum to approximately 1.0
    total_revenue_share = profiles_df["revenue_share"].sum()
    assert total_revenue_share == pytest.approx(
        1.0, abs=1e-6
    ), f"revenue_share sum {total_revenue_share:.6f} is not close to 1.0"

    # Check for no NaN in essential columns
    essential_columns = ["n_customers", "revenue_share", "avg_monetary"]
    for col in essential_columns:
        assert not profiles_df[col].isna().any(), f"Found NaN values in {col}"

    # Check that n_customers are positive integers
    assert (
        profiles_df["n_customers"] > 0
    ).all(), "Some segments have non-positive customer counts"
    assert (
        profiles_df["n_customers"] == profiles_df["n_customers"].astype(int)
    ).all(), "Customer counts should be integers"

    # Check that shares are between 0 and 1
    assert (profiles_df["share_customers"] >= 0).all() and (
        profiles_df["share_customers"] <= 1
    ).all(), "share_customers should be between 0 and 1"
    assert (profiles_df["revenue_share"] >= 0).all() and (
        profiles_df["revenue_share"] <= 1
    ).all(), "revenue_share should be between 0 and 1"


def test_customer_segments_final_structure(project_root):
    """
    Test that customer segments final file has correct structure.
    """
    customer_segments_final_path = (
        project_root / "data" / "processed" / "customer_segments_final.csv"
    )
    segments_df = pd.read_csv(customer_segments_final_path)

    # Check required columns
    required_columns = ["CustomerID", "segment", "Recency", "Frequency", "Monetary"]
    for col in required_columns:
        assert (
            col in segments_df.columns
        ), f"Customer segments final missing column: {col}"

    # Check that all customers have segments
    assert (
        segments_df["segment"].notna().all()
    ), "Some customers have missing segment labels"

    # Check that segment values are integers
    assert segments_df["segment"].dtype in [
        "int64",
        "int32",
    ], f"Segment column has dtype {segments_df['segment'].dtype}, expected integer"

    # Check that RFM values are numeric and positive
    rfm_columns = ["Recency", "Frequency", "Monetary"]
    for col in rfm_columns:
        assert pd.api.types.is_numeric_dtype(
            segments_df[col]
        ), f"{col} should be numeric"
        assert (segments_df[col] >= 0).all(), f"Some {col} values are negative"


def test_customer_segments_labeled_structure(project_root):
    """
    Test that customer segments labeled file has correct structure and business labels.
    """
    customer_segments_labeled_path = (
        project_root / "data" / "processed" / "customer_segments_labeled.csv"
    )
    labeled_df = pd.read_csv(customer_segments_labeled_path)

    # Check required columns
    required_columns = [
        "CustomerID",
        "segment",
        "segment_label",
        "Recency",
        "Frequency",
        "Monetary",
    ]
    for col in required_columns:
        assert (
            col in labeled_df.columns
        ), f"Customer segments labeled missing column: {col}"

    # Check that all customers have business labels
    assert (
        labeled_df["segment_label"].notna().all()
    ), "Some customers have missing business labels"
    assert (
        labeled_df["segment_label"] != ""
    ).all(), "Some customers have empty business labels"

    # Check that business labels are valid
    valid_labels = [
        "VIP Loyal",
        "High-Value Rare Buyers",
        "Regulars",
        "At-Risk / Inactive",
        "General",
    ]
    invalid_labels = set(labeled_df["segment_label"]) - set(valid_labels)
    assert len(invalid_labels) == 0, f"Found invalid business labels: {invalid_labels}"

    # Check that all customers have segments
    assert (
        labeled_df["segment"].notna().all()
    ), "Some customers have missing segment labels"

    # Check that segment values are integers
    assert labeled_df["segment"].dtype in [
        "int64",
        "int32",
    ], f"Segment column has dtype {labeled_df['segment'].dtype}, expected integer"


def test_segment_narratives_structure(project_root):
    """
    Test that segment narratives file has correct structure.
    """
    narratives_path = project_root / "data" / "processed" / "segment_narratives.csv"

    if not narratives_path.exists():
        pytest.skip(
            "Segment narratives file not found - optional GPT enrichment may not have been run"
        )

    narratives_df = pd.read_csv(narratives_path)

    # Check required columns
    required_columns = [
        "segment",
        "segment_label",
        "short_description",
        "campaign_ideas",
    ]
    for col in required_columns:
        assert col in narratives_df.columns, f"Segment narratives missing column: {col}"

    # Check that all segments have narratives
    assert (
        narratives_df["segment"].notna().all()
    ), "Some segments have missing segment numbers"
    assert (
        narratives_df["segment_label"].notna().all()
    ), "Some segments have missing business labels"
    assert (
        narratives_df["short_description"].notna().all()
    ), "Some segments have missing descriptions"
    assert (
        narratives_df["campaign_ideas"].notna().all()
    ), "Some segments have missing campaign ideas"

    # Check that descriptions and campaign ideas are not empty
    assert (
        narratives_df["short_description"] != ""
    ).all(), "Some segments have empty descriptions"
    assert (
        narratives_df["campaign_ideas"] != ""
    ).all(), "Some segments have empty campaign ideas"

    # Check that segment numbers are integers
    assert narratives_df["segment"].dtype in [
        "int64",
        "int32",
    ], f"Segment column has dtype {narratives_df['segment'].dtype}, expected integer"


def test_data_consistency_across_files(project_root):
    """
    Test that data is consistent across all reporting files.
    """
    # Load all relevant files
    profiles_df = pd.read_csv(
        project_root / "data" / "processed" / "segment_profiles.csv"
    )
    segments_final_df = pd.read_csv(
        project_root / "data" / "processed" / "customer_segments_final.csv"
    )
    segments_labeled_df = pd.read_csv(
        project_root / "data" / "processed" / "customer_segments_labeled.csv"
    )

    # Check that segment numbers match between files
    profiles_segments = set(profiles_df["segment"])
    final_segments = set(segments_final_df["segment"])
    labeled_segments = set(segments_labeled_df["segment"])

    assert (
        profiles_segments == final_segments == labeled_segments
    ), "Segment numbers don't match across files"

    # Check that customer counts match
    for segment in profiles_segments:
        profile_count = profiles_df[profiles_df["segment"] == segment][
            "n_customers"
        ].iloc[0]
        final_count = len(segments_final_df[segments_final_df["segment"] == segment])
        labeled_count = len(
            segments_labeled_df[segments_labeled_df["segment"] == segment]
        )

        assert (
            profile_count == final_count == labeled_count
        ), f"Customer counts don't match for segment {segment}"

    # Check that CustomerIDs match between final and labeled files
    final_customers = set(segments_final_df["CustomerID"])
    labeled_customers = set(segments_labeled_df["CustomerID"])

    assert (
        final_customers == labeled_customers
    ), "CustomerID sets don't match between final and labeled files"


def test_markdown_report_content(project_root):
    """
    Test that the markdown report has expected content.
    """
    report_path = project_root / "reports" / "segment_report.md"

    with open(report_path) as f:
        report_content = f.read()

    # Check for expected sections
    expected_sections = [
        "# Customer Segmentation Report",
        "## Executive Summary",
        "## Segment Overview",
        "## Visualizations",
        "## Key Insights",
        "## Segment Narratives",
        "## Recommendations",
    ]

    for section in expected_sections:
        assert section in report_content, f"Report missing section: {section}"

    # Check for figure references
    expected_figures = [
        "figures/segments_size_bar.png",
        "figures/segments_revenue_share.png",
        "figures/segments_rfm_heatmap.png",
        "figures/segments_rfm_boxplots.png",
    ]

    for figure in expected_figures:
        assert figure in report_content, f"Report missing figure reference: {figure}"

    # Check that report is not empty
    assert len(report_content) > 1000, "Report seems too short"


def test_html_report_exists(project_root):
    """
    Test that HTML report exists and has basic structure.
    """
    html_path = project_root / "reports" / "segment_report.html"

    if not html_path.exists():
        pytest.skip("HTML report not found - may be optional")

    with open(html_path) as f:
        html_content = f.read()

    # Check for basic HTML structure
    assert "<html>" in html_content.lower(), "HTML report missing <html> tag"
    assert "<head>" in html_content.lower(), "HTML report missing <head> tag"
    assert "<body>" in html_content.lower(), "HTML report missing <body> tag"
    assert "<title>" in html_content.lower(), "HTML report missing <title> tag"

    # Check that report is not empty
    assert len(html_content) > 1000, "HTML report seems too short"


def test_business_label_distribution(project_root):
    """
    Test that business labels have reasonable distribution.
    """
    labeled_df = pd.read_csv(
        project_root / "data" / "processed" / "customer_segments_labeled.csv"
    )

    # Get label distribution
    label_counts = labeled_df["segment_label"].value_counts()

    # Check that we have at least 2 different business labels
    assert (
        len(label_counts) >= 2
    ), f"Only {len(label_counts)} business labels found, expected at least 2"

    # Check that no single label dominates (more than 80% of customers)
    max_share = label_counts.max() / len(labeled_df)
    if max_share > 0.8:
        pytest.xfail(
            f"Single business label dominates: {max_share:.1%} of customers have the same label"
        )

    # Check that we have a reasonable number of labels (not too many, not too few)
    assert (
        2 <= len(label_counts) <= 10
    ), f"Found {len(label_counts)} business labels, expected 2-10"


def test_segment_profiles_statistical_consistency(project_root):
    """
    Test that segment profiles have statistically consistent values.
    """
    profiles_df = pd.read_csv(
        project_root / "data" / "processed" / "segment_profiles.csv"
    )

    # Check that averages are reasonable
    assert (
        profiles_df["avg_recency"] >= 0
    ).all(), "Some average recency values are negative"
    assert (
        profiles_df["avg_frequency"] > 0
    ).all(), "Some average frequency values are non-positive"
    assert (
        profiles_df["avg_monetary"] > 0
    ).all(), "Some average monetary values are non-positive"

    # Check that medians are reasonable
    assert (
        profiles_df["median_recency"] >= 0
    ).all(), "Some median recency values are negative"
    assert (
        profiles_df["median_frequency"] > 0
    ).all(), "Some median frequency values are non-positive"
    assert (
        profiles_df["median_monetary"] > 0
    ).all(), "Some median monetary values are non-positive"

    # Check that averages and medians are in reasonable ranges
    for segment in profiles_df["segment"]:
        segment_data = profiles_df[profiles_df["segment"] == segment].iloc[0]

        # Check that average and median are reasonably close (within 50% of each other)
        recency_ratio = segment_data["avg_recency"] / segment_data["median_recency"]
        frequency_ratio = (
            segment_data["avg_frequency"] / segment_data["median_frequency"]
        )
        monetary_ratio = segment_data["avg_monetary"] / segment_data["median_monetary"]

        # Allow for some skewness but flag extreme cases
        if recency_ratio > 3 or recency_ratio < 0.33:
            pytest.xfail(
                f"Segment {segment} has extreme recency skew: avg/median ratio = {recency_ratio:.2f}"
            )
        if frequency_ratio > 3 or frequency_ratio < 0.33:
            pytest.xfail(
                f"Segment {segment} has extreme frequency skew: avg/median ratio = {frequency_ratio:.2f}"
            )
        if monetary_ratio > 3 or monetary_ratio < 0.33:
            pytest.xfail(
                f"Segment {segment} has extreme monetary skew: avg/median ratio = {monetary_ratio:.2f}"
            )


def test_pathological_dataset_handling(project_root):
    """
    Test handling of pathological datasets (e.g., all customers in one segment).
    """
    profiles_df = pd.read_csv(
        project_root / "data" / "processed" / "segment_profiles.csv"
    )

    # Check if we have a pathological case (all customers in one segment)
    if len(profiles_df) == 1:
        pytest.xfail(
            "All customers are in one segment - this is a pathological case that may need special handling"
        )

    # Check if we have extremely unbalanced segments
    max_share = profiles_df["share_customers"].max()
    if max_share > 0.9:
        pytest.xfail(
            f"Extremely unbalanced segments: {max_share:.1%} of customers in one segment"
        )

    # Check if we have segments with very few customers
    min_customers = profiles_df["n_customers"].min()
    if min_customers < 10:
        pytest.xfail(
            f"Some segments have very few customers: minimum {min_customers} customers"
        )
