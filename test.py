import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import plotly.express as px
import plotly.graph_objects as go


class PCAAnalysis:
    def __init__(self, df, exclude_columns=None):
        """
        Initialize PCA analysis

        Parameters:
        df (pandas.DataFrame): Input dataframe
        exclude_columns (list): Columns to exclude from PCA
        """
        self.original_df = df
        self.exclude_columns = exclude_columns if exclude_columns else ["Cluster",
                                                                        "Leasing_Friendly"]
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for PCA by excluding specified columns and scaling"""
        # Remove excluded columns
        self.df_pca = self.original_df.drop(
            columns=[col for col in self.exclude_columns if col in self.original_df.columns])

        self.df_pca.set_index("Identifiant_DCR_Personne", inplace=True)

        # Scale the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.df_pca)
        self.scaled_df = pd.DataFrame(self.scaled_data, columns=self.df_pca.columns)

        # Perform PCA
        self.pca = PCA()
        self.pca_result = self.pca.fit_transform(self.scaled_data)

    def plot_eigenvalues(self, ylim=(0, 10), n_components=50):
        """Plot eigenvalues explanation"""
        explained_variance_ratio = self.pca.explained_variance_ratio_[:n_components]
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_components + 1), explained_variance_ratio * 100, 'bo-')
        plt.plot(range(1, n_components + 1), cumulative_variance_ratio * 100, 'ro-')
        plt.ylim(ylim)
        plt.xlabel('Components')
        plt.ylabel('Explained variance ratio (%)')
        plt.title('Scree Plot with Explained Variance Ratio')
        plt.legend(['Individual variance', 'Cumulative variance'])
        plt.grid(True)
        return fig

    def get_feature_contribution(self):
        """Get feature contributions to principal components"""
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i + 1}' for i in range(self.pca.n_components_)],
            index=self.df_pca.columns
        )
        return loadings

    def plot_variable_contribution(self, n_variables=30):
        """Plot variable contributions to first two principal components"""
        loadings = self.get_feature_contribution()

        # Calculate contribution to PC1 and PC2
        contrib = pd.DataFrame({
            'PC1': loadings['PC1'] ** 2,
            'PC2': loadings['PC2'] ** 2
        })

        # Sort by total contribution
        contrib['total'] = contrib['PC1'] + contrib['PC2']
        contrib_sorted = contrib.sort_values('total', ascending=False).head(n_variables)

        fig = plt.figure(figsize=(12, 8))
        plt.scatter(contrib_sorted['PC1'], contrib_sorted['PC2'])

        # Add variable names as annotations
        for idx, row in contrib_sorted.iterrows():
            plt.annotate(idx, (row['PC1'], row['PC2']))

        plt.xlabel('Contribution to PC1')
        plt.ylabel('Contribution to PC2')
        plt.title('Variable Contributions to PC1 and PC2')
        plt.grid(True)
        return fig

    def plot_correlation_matrix(self, variables=None, start_idx=0, end_idx=30):
        """Plot correlation matrix for selected variables"""
        if variables is None:
            variables = self.df_pca.columns[start_idx:end_idx]

        corr_matrix = self.df_pca[variables].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()

    def plot_pca_scatter(self, color_by=None):
        """Plot PCA scatter plot with optional coloring"""
        if color_by and color_by in self.original_df.columns:
            color_values = self.original_df[color_by]
        else:
            color_values = None

        fig = px.scatter(
            x=self.pca_result[:, 0],
            y=self.pca_result[:, 1],
            color=color_values,
            labels={'x': 'PC1', 'y': 'PC2'},
            title='PCA Scatter Plot'
        )
        return fig

    def plot_focused_contribution(self, variable_list, title="Focused Contribution"):
        """Plot contribution of specific variables"""
        loadings = self.get_feature_contribution()
        focused_loadings = loadings.loc[variable_list, ['PC1', 'PC2']]

        fig = plt.figure(figsize=(12, 8))
        plt.scatter(focused_loadings['PC1'], focused_loadings['PC2'])

        for idx, row in focused_loadings.iterrows():
            plt.annotate(idx, (row['PC1'], row['PC2']))

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(title)
        plt.grid(True)
        return fig


# Example usage:
def main():
    # Assuming you're using dataiku
    import dataiku

    # Read the dataset
    df = dataiku.Dataset("Prospect_Clustering_BoxCox").get_dataframe()

    # Initialize PCA analysis
    pca_analysis = PCAAnalysis(df)

    # Plot eigenvalues
    fig_eigen = pca_analysis.plot_eigenvalues()
    plt.show()

    # Plot variable contributions
    fig_contrib = pca_analysis.plot_variable_contribution()
    plt.show()

    # Plot correlation matrix for first 30 variables
    pca_analysis.plot_correlation_matrix()
    plt.show()

    # Plot PCA scatter colored by cluster
    fig_scatter = pca_analysis.plot_pca_scatter(color_by='Cluster')
    fig_scatter.show()

    # Plot BNP focused contribution
    bnp_variables = [
        "ENCOURS_OUI_LEASING_E2", "ENCOURS_BNPP_LEASING_GA",
        "ENCOURS_OUI_LEASING_GA", "ENCOURS_BNPP_HT_E2",
        "ENCOURS_BNPP_HT_GA", "ENCOURS_OUI_HT_E2",
        "ENCOURS_OUI_HT_GA"
    ]
    fig_bnp = pca_analysis.plot_focused_contribution(bnp_variables, "BNP Focused Contribution")
    plt.show()

    # Plot financial focused contribution
    financial_variables = [
        "Actif_brut_2017_4", "Actif_brut_2016_4",
        "Amortissement_2017_4", "Amortissement_2016_4",
        "Cap_2017_4", "Cap_2016_4",
        "Actif_net_2017_4", "Actif_net_2016_4",
        "CA_2017_4", "CA_2016_4",
        "Capex_2017_4", "Capex_2016_4"
    ]
    fig_financial = pca_analysis.plot_focused_contribution(financial_variables, "Financial Focused Contribution")
    plt.show()

    # Save results if using dataiku
    output_df = pd.DataFrame(pca_analysis.pca_result,
                             columns=[f'PC{i + 1}' for i in range(pca_analysis.pca.n_components_)])
    dataiku.Dataset("Prospect_Clustering_PCA_Explain").write_with_schema(output_df)


if __name__ == "__main__":
    main()
