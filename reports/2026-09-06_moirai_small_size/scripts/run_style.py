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
    "cf412oe2": ("#3f51b5", OURS),
    "cf412om2": ("#ffb300", OURS),
    "cf412oa2": ("#800000", OURS),
    "cf412ow2": ("#b8860b", OURS),
    "cf412or2": ("#9acd32", OURS),
    "cf412ol2": ("#1b7837", OURS),
    "cf412bm": ("#8a2be2", OURS),
    "cf412bw": ("#ff00ff", OURS),
    "cf412al": ("#fa8072", OURS),
    "cf419ms": ("#556b2f", OURS),
    "cf415_moirai": ("#8c564b", MOIRAI),
    "cf415_moirai_native": ("#8c564b", ":"),
    "cf419_moirai_native": ("#9a9a00", MOIRAI),
    "cf421z_moirai_native": ("#c51b7d", MOIRAI),
    "cf421f_moirai_native": ("#d62728", MOIRAI),
    "cf421fb_moirai_native": ("#d4a017", MOIRAI),
    "cf421n_moirai_native": ("#e7298a", MOIRAI),
    "cf421ew_moirai_native": ("#000080", MOIRAI),
}


# run: a 3-letter code (owner, 10-02). Every legend shows it in front of the
# run's label, so that the owner and the agents name a run the same way.
# Ours, one patch size and EWMA: ABC is the owner's name for the lr 5.6e-5
# run. Ours + patch sizes: O, then the recipe (M Moirai, C cyan, E cyan with
# EWMA), then B for the loss bug or F for the loss fixed. Moirai: M, then O
# for its own head or P for patch heads, then the variant.
CODE = {
    P + "_lr10x": "ABC", P + "_lr10xb": "TWN", P + "_lr30x": "LOW", P + "_lr100x": "MIN",
    P + "_cos665k": "LNG", P + "_cos200k": "CYN", "cf419_cos200k": "BLK", "b2cos": "WDT",
    "cf412om": "OMB", "cf412oc": "OCB", "cf412oc2": "OCF", "cf412oe2": "OEF", "cf412om2": "OMF", "cf412oa2": "OAF", "cf412ow2": "OWF", "cf412or2": "OWR", "cf412ol2": "OWL", "cf412bm": "OBM", "cf412bw": "OBW", "cf412al": "OAL", "cf419ms": "BMS",
    "cf415_moirai": "MSH", "cf415_moirai_native": "MOO", "cf419_moirai_native": "MON",
    "cf421f_moirai_native": "MPM", "cf421ew_moirai_native": "MPE", "cf421n_moirai_native": "MPR",
    "cf421z_moirai_native": "MPZ", "cf421fb_moirai_native": "MPG",
}
assert len(set(CODE.values())) == len(CODE), "two runs share a code"


def tagged(run, label):
    """A legend label: the run's code in bold, then the label."""
    return rf"$\mathbf{{{CODE[run]}}}$  {label}"


def colour(run):
    return STYLE[run][0]


def line(run):
    return STYLE[run][1]
