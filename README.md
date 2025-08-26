# Customer Segmentation Project

## Project Description

This project implements customer segmentation using RFM (Recency, Frequency, Monetary) analysis and machine learning clustering techniques. The goal is to identify distinct customer segments to enable targeted marketing strategies and improve customer relationship management.

### What is Customer Segmentation?

Customer segmentation is the process of dividing customers into groups based on shared characteristics. This project focuses on behavioral segmentation using RFM analysis, which is a proven methodology for understanding customer value and behavior patterns.

## Dataset

### Primary Dataset
- **Kaggle Dataset**: [Online Retail Dataset (UCI)](https://www.kaggle.com/datasets/mathchi/online-retail-ii-data-set-from-real-ecommerce-company)
- **Source**: UCI Machine Learning Repository
- **Description**: Online retail transaction data from a UK-based e-commerce company

### Simulated Dataset
Since the Kaggle dataset may not always be available, this project includes a reproducible simulated dataset with the following characteristics:
- **Columns**: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice
- **Time Period**: 2 years of transaction data (2022-2024)
- **Customers**: 1,000 unique customers
- **Transactions**: 5,000 total transactions
- **Reproducibility**: Uses `np.random.seed(42)` for consistent results

## Methodology

### Step 1: RFM Analysis
RFM (Recency, Frequency, Monetary) analysis is a marketing technique used to determine customer value by examining:

1. **Recency (R)**: How recently a customer has made a purchase
   - Measured in days since last purchase
   - Lower values indicate more recent activity

2. **Frequency (F)**: How often a customer makes a purchase
   - Total number of transactions
   - Higher values indicate more frequent purchases

3. **Monetary (M)**: How much money a customer spends
   - Total spending amount across all transactions
   - Higher values indicate higher customer value

### Step 2: Data Scaling
Before applying clustering algorithms, RFM metrics are standardized using `StandardScaler` from scikit-learn. This is crucial because:

- **Different Scales**: RFM metrics have vastly different scales (e.g., Recency: 1-723 days, Monetary: $214-$108,384)
- **Clustering Performance**: Algorithms like K-means are sensitive to feature scales
- **Fair Comparison**: Ensures all features contribute equally to the clustering process

**Scaling Process**:
- **StandardScaler**: Transforms features to have mean=0 and standard deviation=1
- **Formula**: `z = (x - μ) / σ` where μ is the mean and σ is the standard deviation
- **Result**: All features are on the same scale for fair clustering

**Example Scaled Data**:
```
CustomerID  Recency_Scaled  Frequency_Scaled  Monetary_Scaled
CUST_0001       -0.871         -0.228        -1.214
CUST_0002        2.568         -0.706        -0.355
CUST_0003       -0.793          0.729         0.598
CUST_0004        0.704         -0.706        -0.722
```

**Testing**: A test script (`src/test_scaling.py`) verifies that scaled features have mean ≈ 0 and std ≈ 1 within tolerance.

### Clustering Approach
The project uses machine learning clustering algorithms to group customers based on their RFM scores:
- **K-means Clustering**: Primary segmentation method
- **Hierarchical Clustering**: Alternative approach for validation
- **Silhouette Analysis**: Optimal cluster number determination

## Folder Structure

```
customer-segmentation/
│
├── data/
│   ├── raw/                    # Original/simulated transaction data
│   │   └── sample_transactions.csv
│   └── processed/              # Cleaned and processed data
│       ├── rfm_table.csv       # RFM metrics per customer
│       └── rfm_scaled.csv      # Scaled RFM data for clustering
├── notebooks/                  # Jupyter notebooks for analysis
│   └── 01_rfm_analysis.ipynb
├── src/                        # Python source code
│   ├── rfm_analysis.py         # Step 1: RFM analysis
│   ├── rfm_scaling.py          # Step 2: Data scaling
│   └── test_scaling.py         # Scaling verification tests
├── streamlit_app/              # Interactive web application
├── reports/                    # Generated reports and visualizations
│   └── figures/
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Tech Stack

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations

### Development Tools
- **jupyter**: Interactive notebooks
- **streamlit**: Web application framework
- **openpyxl**: Excel file handling
- **xlrd**: Legacy Excel file support

### Optional Integrations
- **OpenAI API**: AI-powered insights and commentary
- **plotly-dash**: Advanced interactive dashboards

## Usage

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd customer-segmentation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the RFM analysis**:
   ```bash
   python src/rfm_analysis.py
   ```

4. **Scale the RFM data**:
   ```bash
   python src/rfm_scaling.py
   ```

5. **Test the scaling**:
   ```bash
   python src/test_scaling.py
   ```

6. **Open Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/01_rfm_analysis.ipynb
   ```

### Running the Analysis

#### Option 1: Python Scripts (Step by Step)
```bash
# Step 1: RFM Analysis
python src/rfm_analysis.py

# Step 2: Data Scaling
python src/rfm_scaling.py

# Step 3: Test Scaling
python src/test_scaling.py
```

#### Option 2: Jupyter Notebook
```bash
cd notebooks
jupyter notebook 01_rfm_analysis.ipynb
```

### Expected Output
- Raw transaction data saved to `data/raw/sample_transactions.csv`
- RFM metrics table saved to `data/processed/rfm_table.csv`
- Scaled RFM data saved to `data/processed/rfm_scaled.csv`
- Console output showing:
  - Dataset generation statistics
  - Data cleaning results
  - RFM metrics for first 10 customers
  - Scaling statistics (mean ≈ 0, std ≈ 1)
  - Summary statistics

## Future Work

### Phase 2: Advanced Segmentation
- [ ] Implement K-means clustering on RFM scores
- [ ] Add hierarchical clustering for validation
- [ ] Create segment profiles and characteristics
- [ ] Develop segment-specific marketing strategies

### Phase 3: Interactive Dashboard
- [ ] Build Streamlit web application
- [ ] Create interactive visualizations
- [ ] Add real-time data upload capability
- [ ] Implement segment comparison tools

### Phase 4: SaaS Integration
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] API development for external access
- [ ] Automated reporting and alerts
- [ ] Multi-tenant architecture

### Phase 5: AI Enhancement
- [ ] OpenAI API integration for insights
- [ ] Automated segment naming and description
- [ ] Predictive customer behavior modeling
- [ ] Natural language query interface

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for the original dataset
- Kaggle community for dataset hosting and discussions
- RFM analysis methodology pioneers in marketing science

---

**Note**: This project is designed for educational and research purposes. When using with real customer data, ensure compliance with data protection regulations (GDPR, CCPA, etc.).
