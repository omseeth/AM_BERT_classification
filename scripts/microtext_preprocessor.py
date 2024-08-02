"""
This script helps with preprocessing data for training and test sets from
the Argumentative Microtext Corpus Part One and Part Two (references below).

Terminology:

EDU = elementary discourse unit
ADU = argumentation discourse unit

Example of annotated text sample:

## Annotation after corpus is loaded with custom script: ##
edus     {'e1': 'Hunting is good for the environment', ...
adus     {'a1': 'pro', 'a2': 'pro', 'a3': 'pro', 'a4': ...
edges    [[c7, e1, a1, seg], [c8, e2, a2, seg], [c9, e3...

label types:
pro - Proponent ADU "for" sth.
opp - Opponent ADU "against" sth.

edge types:
seg - segment  # Not all EDUs are ADUs
sup - support
exa - example
reb - rebuttal
und - undercut
add - addition

## Text: ##
[Hunting is good for the environment because o...]


For reference, please consider:

Part One:
Andreas Peldszus and Manfred Stede. An annotated corpus of argumentative
microtexts. In D. Mohammed, and M. Lewinski, editors, Argumentation and
Reasoned Action - Proc. of the 1st European Conference on Argumentation,
Lisbon, 2015. College Publications, London, 2016

Part Two:
Maria Skeppstedt, Andreas Peldszus and Manfred Stede. More or less
controlled elicitation of argumentative text: Enlarging a microtext corpus via
crowdsourcing. In Proc. 5th Workshop in Argumentation Mining (at EMNLP),
Brussels, 2018

last change
OMS 22.07.2024
"""
from corpus_loader import CorpusLoader

import copy
import pandas as pd


class MicrotextPropositionPrepper:
    """
    Extracts the component labels and corresponding text instances from
    corpus.
    """

    def instance_extraction(self, proposition_df):
        """
        Extracts labels and corresponding ADUs from DataFrames.

        Parameter:
            component_df, DataFrame: with EDU, ADU, and edge information

        Return:
            DataFrame: with two columns corresponding to ADUS and their labels
        """
        labels = []
        instances = []

        for index, row in proposition_df.iterrows():
            edus = row["edus"]
            adus = row["adus"]
            edges = row["edges"]

            for edge in edges:
                if edge[3] == "seg":
                    instances.append(edus[edge[1]])
                    del edus[edge[1]]  # Remove edu segment from dictionary
                    labels.append(adus[edge[2]])

            for edu in edus:  # Edus that were not assigned an Adu receive None
                instances.append(edu)
                labels.append("None")

        training_data = zip(labels, instances)
        df = pd.DataFrame(training_data, columns=["label", "text"])

        return df


class MicrotextRelationPrepper:
    """
    Extracts the relation labels and corresponding text instances from
    corpus.
    """

    def pair_extraction(self,
                        relation_df,
                        directed=True,
                        window=False,
                        window_size=1):
        """
        Extracts labels and corresponding ordered pairs from each annotation
        of document.

        Parameters:
            relation_df, DataFrame: with EDUs, ADUs, and edges for relations
            directed, bool: if True, only assign edge value to ordered pairs,
                            if False, pairs and their mirrors are assigned a
                            value
            window, bool: if True, only pairs from window will be considered
            window_size, int: size if window for each EDU to the left and
                              right

        Return:
            DataFrame: with relation labels for pair of ADUs
        """
        labels = []
        pairs = []

        for index, row in relation_df.iterrows():
            edus = row["edus"]
            edges = row["edges"]

            segments = {}
            relations = {"sup": [], "exa": [], "reb": [], "und": [], "add": []}

            for edge in edges:
                if edge[3] == "seg":
                    segments.update({edge[2]: edus[edge[1]]})

                if edge[3] == "sup":
                    relations["sup"].append([edge[1], edge[2]])

                if edge[3] == "exa":
                    relations["exa"].append([edge[1], edge[2]])

                if edge[3] == "reb":
                    relations["reb"].append([edge[1], edge[2]])

                if edge[3] == "und":
                    relations["und"].append([edge[1], edge[2]])

                if edge[3] == "add":
                    relations["add"].append([edge[1], edge[2]])

            adus_from_text = list(segments.keys())

            if window:
                pairs_window = self.pair_window(adus_from_text, window_size)

                pair_row, label_row = self.pair_labeling(segments,
                                                         relations,
                                                         pairs_window,
                                                         directed)
            else:
                pairs_wo_replacement = self.pair_wo_replacement(adus_from_text)

                pair_row, label_row = self.pair_labeling(segments,
                                                         relations,
                                                         pairs_wo_replacement,
                                                         directed)

            pairs.extend(pair_row)
            labels.extend(label_row)

            # Adding undercutter and addition relations to dataset for
            # corpus statistics
            for relation in relations["und"]:
                pairs.append(relation)
                labels.append("und")

            for relation in relations["add"]:
                pairs.append(relation)
                labels.append("add")

        training_data = zip(labels, pairs)
        df = pd.DataFrame(training_data, columns=["label", "pairs"])

        return df

    def pair_labeling(self, segments, relations, raw_pairs, directed):
        """
        Assigns to all combination of ADU pairs one of the following
        relations: support, example, rebuttal, None.

        ## Directedness: ##

        Directed: When any of the ordered pairs have a relation, then their
        receive a corresponding label.
        Undirected: If relation exists for one tuple, e.g., (a, b),
        then relation will equally be considered for (b, a), and the value for
        both pairs is set as the relation type.

        Parameters:
            segments, dict: ADUs and their corresponding texts (EDUs)
            relations, dict: for each document a dictionary of relations
                             indicated by lists of related ADUs
            raw_pairs, list: with pairs of ADUs (e.g., all combinations of
                             ADUs)
            directed, bool: if True, only assign edge value to ordered pairs,
                            if False, pairs and their mirrors are assigned a
                            value

        Return:
            tuple: of ADU pairs and their annotated relations
        """
        sup = relations["sup"]
        exa = relations["exa"]
        reb = relations["reb"]

        pairs = []
        labels = []

        if raw_pairs:  # Check if any pairs have been handed over
            for pair in raw_pairs:
                if pair in sup:
                    pairs.append((segments[pair[0]], segments[pair[1]]))
                    labels.append("sup")
                    if not directed:
                        pair_mirror = [pair[1], pair[0]]
                        pairs.append([segments[pair[1]], segments[pair[0]]])
                        labels.append("sup")
                        # Mirror pair is assigned same label
                        raw_pairs.remove(pair_mirror)
                elif pair in exa:
                    pairs.append((segments[pair[0]], segments[pair[1]]))
                    labels.append("exa")
                    if not directed:
                        pair_mirror = [pair[1], pair[0]]
                        pairs.append([segments[pair[1]], segments[pair[0]]])
                        labels.append("exa")
                        # Mirror pair is assigned same label
                        raw_pairs.remove(pair_mirror)
                elif pair in reb:
                    pairs.append((segments[pair[0]], segments[pair[1]]))
                    labels.append("reb")
                    if not directed:
                        pair_mirror = [pair[1], pair[0]]
                        pairs.append([segments[pair[1]], segments[pair[0]]])
                        labels.append("reb")
                        # Mirror pair is assigned same label
                        raw_pairs.remove(pair_mirror)
                else:
                    pairs.append((segments[pair[0]], segments[pair[1]]))
                    labels.append("None")

        return pairs, labels

    def pair_wo_replacement(self, segments):
        """
        From the given ADUs each ADU is paired with each other ADU, except
        for itself. The result is a list of ordered pairs sampled without
        replacement.

        For example: permutation of 2 out of 4 should result in 12 pairs.

        4!/(4-2)! = 12

        Parameter:
            segments, list: of ADU segments

        Return:
            list: of lists with all possible combinations of ADUs without
                  replacement
        """
        combinations = []
        for i in segments:
            for j in segments:
                if i != j:
                    combinations.append([i, j])

        return combinations

    def pair_window(self, adus_from_text, window_size):
        assert type(window_size) == int
        combinations = []
        n = len(adus_from_text)

        for i in range(n):
            start = max(0, i - window_size)
            # +1 to include the element at i + window_size
            end = min(n, i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    combinations.append([adus_from_text[i], adus_from_text[j]])

        return combinations


def proposition_statistic(all_annotation):
    """
    Skeppstedt et al. (2017, p. 158):

    "A total of 205 texts had been originally collected, and from these,
    34 were excluded from further consideration".
    """
    components = MicrotextPropositionPrepper()
    #  Pandas has no deepcopy function for nested elements
    copy_annotation = pd.DataFrame(copy.deepcopy(all_annotation.to_dict()))

    edus_n = 0

    for index, row in all_annotation.iterrows():
        edus = row["edus"]
        edus_n += len(edus.items())

    all_set = components.instance_extraction(copy_annotation)

    print(all_set.head())
    print(all_set.tail())

    pro_all = len(all_set[all_set["label"] == "pro"])
    opp_all = len(all_set[all_set["label"] == "opp"])
    none_all = len(all_set[all_set["label"] == "None"])

    print(f"""
        Corpus statistics:

        All documents from corpus: {len(all_annotation)}
        Averages EDUs per document: {edus_n / len(all_annotation)}
        
        ADUs with "opp": {opp_all}
        ADUs with "pro": {pro_all}
        EDUs without label: {none_all}
        EDUs in sum: {pro_all + opp_all + none_all}
        """)


def relation_statistics(all_annotation):
    """
    Skeppstedt et al. (2017, p. 160):

    "The distribution of relation is: convergent support (467); example
    support (23); rebutting attack (137); undercutting attack (77); linked
    support or attack (57); restatement (29)."
    """
    relations = MicrotextRelationPrepper()
    directed = True  # Consider order of relation pairs
    window = True  # If True only pairs within a window are considered
    window_size = 3  # Size of window to the left and right

    all_set = relations.pair_extraction(all_annotation,
                                        directed,
                                        window,
                                        window_size)

    print(all_set.head())
    print(all_set.tail())

    sup_all = len(all_set[all_set["label"] == "sup"])
    exa_all = len(all_set[all_set["label"] == "exa"])
    reb_all = len(all_set[all_set["label"] == "reb"])
    und_all = len(all_set[all_set["label"] == "und"])
    add_all = len(all_set[all_set["label"] == "add"])
    none_all = len(all_set[all_set["label"] == "None"])

    relations_sum = sup_all + exa_all + reb_all

    print(f"""
          Relation statistics:
          
          All relations in corpus: {relations_sum}
          
          Support relations in corpus: {sup_all}
          Example relations in corpus: {exa_all}
          Rebuttal relations in corpus: {reb_all}
          Undercutter relations in corpus: {und_all}
          Linked (add) relations in corpus: {add_all}
          None relations in corpus: {none_all}
          """)


if __name__ == "__main__":
    url_microtext = "https://github.com/peldszus/arg-microtexts/archive/refs/heads/master.zip"
    microtext_save_path = "../data/corpus_2/"
    microtext_path = "../data/corpus_2/arg-microtexts-master/corpus/en/"
    microtext_file_type_ann = ".xml"
    microtext_file_type_text = ".txt"

    corpus_loader = CorpusLoader()
    corpus_loader.download_file_from(url_microtext,
                                     microtext_save_path)

    all_annotation = corpus_loader.load_files_from(microtext_path,
                                                   microtext_file_type_ann)

    proposition_statistic(all_annotation)
    relation_statistics(all_annotation)
