import numpy as np
import numpy.typing as npt


def import_htm_data(material: str):
    import h_transport_materials as htm

    permeabilities = htm.permeabilities.filter(material=f"{material}").filter(
        isotope="h"
    )

    def arrhenius_law(T, A, E):
        k_B = 8.617333262145e-5  # eV/K
        return A * np.exp(-E / (k_B * T))

    all_x_values = []
    all_y_values = []
    labels = []

    for entry in permeabilities:
        if entry.range is None:
            T_bounds = (300, 1200)
        else:
            T_bounds = (entry.range[0].magnitude, entry.range[-1].magnitude)

        x_values = np.geomspace(T_bounds[0], T_bounds[1], num=100)

        y_values = arrhenius_law(
            x_values, entry.pre_exp.magnitude, entry.act_energy.magnitude
        )
        # obtain labels
        label = f"{entry.author} ({entry.year})"
        label = label.capitalize()

        all_x_values.append(x_values)
        all_y_values.append(y_values)
        labels.append(label)

    return all_x_values, all_y_values, labels
