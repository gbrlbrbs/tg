from pyhard.measures import ClassificationMeasures
import pandas as pd

def _match_measure(measure_name: str, classification_measures: ClassificationMeasures):
    match measure_name:
        case 'N1':
            return classification_measures.borderline_points()
        case 'N2':
            return classification_measures.intra_extra_ratio()
        case 'DCP':
            return classification_measures.disjunct_class_percentage()
        case 'TD_P':
            return classification_measures.tree_depth_pruned()
        case 'TD_U':
            return classification_measures.tree_depth_unpruned()
        case 'CL':
            return classification_measures.class_likeliood()
        case 'CLD':
            return classification_measures.class_likeliood_diff()
        case 'LSC':
            return classification_measures.local_set_cardinality()
        case 'LSR':
            return classification_measures.ls_radius()
        case 'Harmfulness':
            return classification_measures.harmfulness()
        case 'Usefulness':
            return classification_measures.usefulness()
        case 'F1':
            return classification_measures.f1()


def calculate_measures(df: pd.DataFrame, target_col, measure_list: list[str]):
    class_measures = ClassificationMeasures(df, target_col=target_col)
    measures_dict = {}
    for measure in measure_list:
        measures_dict[measure] = _match_measure(measure, class_measures)
    return measures_dict