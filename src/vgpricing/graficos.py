"""Estilo comum dos gráficos dos notebooks: fontes legíveis e layout que reserva espaço para rótulos e legendas
(constrained layout do matplotlib). Chame `estilo()` uma vez no início de cada notebook."""
import matplotlib as mpl


def estilo() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
