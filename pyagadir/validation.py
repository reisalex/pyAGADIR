import matplotlib.pyplot as plt
from pyagadir.models import AGADIR
import json
from importlib import resources
from pathlib import Path

def get_package_data_dir():
    """Get the path to the data directory in the installed package."""
    with resources.path('pyagadir', 'data') as data_path:
        return data_path

def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    data_dir = get_package_data_dir()
    figures_dir = data_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    return figures_dir

def plot_ph_helix_content(paper_measured_data_ph, paper_measured_data_helix,
                        paper_predicted_data_ph, paper_predicted_data_helix,
                        pyagadir_predicted_data_helix, peptide, ncap, ccap,
                        ax=None, fig=None):
    """Create a plot comparing measured and predicted helix content vs pH.
    
    Args:
        paper_measured_data_ph: List of pH values from paper measurements
        paper_measured_data_helix: List of helix content values from paper measurements
        paper_predicted_data_ph: List of pH values from paper predictions
        paper_predicted_data_helix: List of helix content values from paper predictions
        pyagadir_predicted_data_helix: List of helix content values from PyAGADIR
        peptide: Peptide sequence
        ncap: N-terminal capping
        ccap: C-terminal capping
        ax: Optional matplotlib axis to plot on. If None, creates new figure.
        fig: Optional matplotlib figure to plot on. If None, creates new figure.

    Returns:
        matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(
        paper_predicted_data_ph,
        paper_predicted_data_helix,
        "o",
        color="white",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=5,
        label="Paper predicted",
    )
    ax.plot(
        paper_measured_data_ph,
        paper_measured_data_helix,
        "o",
        color="black", 
        markersize=5,
        label="Paper measured",
    )
    ax.plot(
        paper_predicted_data_ph,
        pyagadir_predicted_data_helix,
        "o",
        color="orange",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=5,
        label="PyAGADIR",
    )

    # Configure plot
    ax.set_xlim(2.8, 12)
    ax.set_ylim(0, 65)
    ax.set_xlabel("pH")
    ax.set_ylabel("Helix content (%)")
    ax.legend()

    # Set title with caps
    title = ""
    if ncap == "Z":
        title += "Ac-"
    elif ncap == "X":
        title += "Succ-"
    title += peptide
    if ccap == "B":
        title += "-Am"
    ax.set_title(title, fontsize=12)
    
    return fig, ax

def reproduce_figure_3b(method="1s"):
    """
    Reproduce figure 3b from the AGADIR paper.
    """
    # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'figure_3_data.json'
    figures_dir = ensure_figures_dir()

    # Load validation data
    with open(validation_file, "r") as f:
        data = json.load(f)

    peptide = data["figure3b"]["peptide"]
    ncap = data["figure3b"]["ncap"]
    ccap = data["figure3b"]["ccap"]

    paper_measured_data_ph = data["figure3b"]["measured_data_ph"]
    paper_measured_data_helix = data["figure3b"]["measured_data_helix"]
    paper_predicted_data_ph = data["figure3b"]["predicted_data_ph"]
    paper_predicted_data_helix = data["figure3b"]["predicted_data_helix"]

    # AGADIR results
    pyagadir_predicted_data_helix = []
    for ph in paper_predicted_data_ph:
        model = AGADIR(method=method, T=0.0, M=0.1, pH=ph)
        result = model.predict(peptide, ncap=ncap, ccap=ccap)
        pyagadir_predicted_data_helix.append(result.get_percent_helix())
        
    # Plot
    fig, ax = plot_ph_helix_content(paper_measured_data_ph, paper_measured_data_helix,
                                    paper_predicted_data_ph, paper_predicted_data_helix,
                                    pyagadir_predicted_data_helix, peptide, ncap, ccap)

    # Save figure
    fig.savefig(figures_dir / "figure_3b.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def reproduce_figure_4(method="1s"):
    """
    Reproduce figure 4 from the AGADIR paper.
    """
    # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'figure_4_data.json'
    figures_dir = ensure_figures_dir()

    # Load validation data
    with open(validation_file, "r") as f:
        data = json.load(f)

    # Create figure
    fig, axs = plt.subplots(4, 2, figsize=(10, 16))
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot each figure
    for i, (figname, ax) in enumerate(zip(data.keys(), axs.flatten())):
        fig_data = data[figname]
        peptide = fig_data["peptide"]
        ncap = fig_data["ncap"]
        ccap = fig_data["ccap"]

        paper_measured_data_ph = fig_data["measured_data_ph"]
        paper_measured_data_helix = fig_data["measured_data_helix"]
        paper_predicted_data_ph = fig_data["predicted_data_ph"]
        paper_predicted_data_helix = fig_data["predicted_data_helix"]

        pyagadir_predicted_data_helix = []
        for ph in paper_predicted_data_ph:
            model = AGADIR(method=method, T=0.0, M=0.1, pH=ph)
            result = model.predict(peptide, ncap=ncap, ccap=ccap)
            pyagadir_predicted_data_helix.append(result.get_percent_helix())
            

        _, ax = plot_ph_helix_content(paper_measured_data_ph, paper_measured_data_helix,
                                        paper_predicted_data_ph, paper_predicted_data_helix,
                                        pyagadir_predicted_data_helix, peptide, ncap, ccap,
                                        ax=ax)


    fig.savefig(figures_dir / "figure_4.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    method = "r"
    reproduce_figure_3b(method=method)
    reproduce_figure_4(method=method)
