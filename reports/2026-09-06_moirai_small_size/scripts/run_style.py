"""One colour and one line style per run, for every figure of the report.

Ours (our contrastive model) draws solid lines, and Moirai (our copy of
Moirai) draws dashed lines. A run keeps its colour in every figure."""

OURS, MOIRAI = "-", "--"
P = "k3_r100_09_lr56_fix09_dec10k"

# run: (colour, line style)
STYLE = {
    P + "_lr10x": ("#1f77b4", OURS),
    P + "_lr10xb": ("#7fb8e0", OURS),
    P + "_lr30x": ("#7f7f7f", OURS),
    P + "_lr100x": ("#ff9f40", OURS),
    P + "_cos665k": ("#9467bd", OURS),
    P + "_cos200k": ("#17becf", OURS),
    "b2cos": ("#e377c2", OURS),
    "cf419_cos200k": ("#1a1a1a", OURS),
    "cf412om": ("#00897b", OURS),
    "cf412oc": ("#7a0177", OURS),
    "cf412oc2": ("#d95f02", OURS),
    "cf415_moirai": ("#8c564b", MOIRAI),
    "cf415_moirai_native": ("#8c564b", ":"),
    "cf419_moirai_native": ("#9a9a00", MOIRAI),
    "cf421z_moirai_native": ("#c51b7d", MOIRAI),
    "cf421f_moirai_native": ("#d62728", MOIRAI),
    "cf421fb_moirai_native": ("#d4a017", MOIRAI),
    "cf421n_moirai_native": ("#e7298a", MOIRAI),
    "cf421ew_moirai_native": ("#000080", MOIRAI),
}


def colour(run):
    return STYLE[run][0]


def line(run):
    return STYLE[run][1]
