def explain_result(selected_individual):
    explanations = {
        "cap-shape": {
            "b": "bell",
            "c": "conical",
            "x": "convex",
            "f": "flat",
            "k": "knobbed",
            "s": "sunken"
        },
        "cap-surface": {
            "f": "fibrous",
            "g": "grooves",
            "y": "scaly",
            "s": "smooth"
        },
        "cap-color": {
            "n": "brown",
            "b": "buff",
            "c": "cinnamon",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow"
        },
        "bruises": {
            "t": "bruises",
            "f": "no bruises"
        },
        "odor": {
            "a": "almond",
            "l": "anise",
            "c": "creosote",
            "y": "fishy",
            "f": "foul",
            "m": "musty",
            "n": "none",
            "p": "pungent",
            "s": "spicy"
        },
        "gill-attachment": {
            "a": "attached",
            "d": "descending",
            "f": "free",
            "n": "notched"
        },
        "gill-spacing": {
            "c": "close",
            "w": "crowded",
            "d": "distant"
        },
        "gill-size": {
            "b": "broad",
            "n": "narrow"
        },
        "gill-color": {
            "k": "black",
            "n": "brown",
            "b": "buff",
            "h": "chocolate",
            "g": "gray",
            "r": "green",
            "o": "orange",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow"
        },
        "stalk-shape": {
            "e": "enlarging",
            "t": "tapering"
        },
        "stalk-root": {
            "b": "bulbous",
            "c": "club",
            "u": "cup",
            "e": "equal",
            "z": "rhizomorphs",
            "r": "rooted",
            "?": "missing"
        },
        "stalk-surface-above-ring": {
            "f": "fibrous",
            "y": "scaly",
            "k": "silky",
            "s": "smooth"
        },
        "stalk-surface-below-ring": {
            "f": "fibrous",
            "y": "scaly",
            "k": "silky",
            "s": "smooth"
        },
        "stalk-color-above-ring": {
            "n": "brown",
            "b": "buff",
            "c": "cinnamon",
            "g": "gray",
            "o": "orange",
            "p": "pink",
            "e": "red",
            "w": "white",
            "y": "yellow"
        },
        "stalk-color-below-ring": {
            "n": "brown",
            "b": "buff",
            "c": "cinnamon",
            "g": "gray",
            "o": "orange",
            "p": "pink",
            "e": "red",
            "w": "white",
            "y": "yellow"
        },
        "veil-type": {
            "p": "partial",
            "u": "universal"
        },
        "veil-color": {
            "n": "brown",
            "o": "orange",
            "w": "white",
            "y": "yellow"
        },
        "ring-number": {
            "n": "none",
            "o": "one",
            "t": "two"
        },
        "ring-type": {
            "c": "cobwebby",
            "e": "evanescent",
            "f": "flaring",
            "l": "large",
            "n": "none",
            "p": "pendant",
            "s": "sheathing",
            "z": "zone"
        },
        "spore-print-color": {
            "k": "black",
            "n": "brown",
            "b": "buff",
            "h": "chocolate",
            "r": "green",
            "o": "orange",
            "u": "purple",
            "w": "white",
            "y": "yellow"
        },
        "population": {
            "a": "abundant",
            "c": "clustered",
            "n": "numerous",
            "s": "scattered",
            "v": "several",
            "y": "solitary"
        },
        "habitat": {
            "g": "grasses",
            "l": "leaves",
            "m": "meadows",
            "p": "paths",
            "u": "urban",
            "w": "waste",
            "d": "woods"
        }
    }

    explanation = ""
    for feature, value in selected_individual.items():
        if feature in explanations:
            explanation += f"{feature}: {explanations[feature].get(value, value)}\n"
        else:
            explanation += f"{feature}: {value}\n"

    return explanation