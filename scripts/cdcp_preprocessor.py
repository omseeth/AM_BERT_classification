"""
This script helps with preprocessing data for training and test sets from the
Cornell eRulemaking Corpus – CDCP (reference below).

Terminology:

EDU = elementary discourse unit
ADU = argumentation discourse unit

In CDCP, EDUs = ADUs.

Example of annotated text sample:

## Annotation after corpus is loaded with custom script: ##
evidences                                     []
prop_labels               [value, value, policy]
prop_offsets    [[0, 78], [78, 242], [242, 391]]
reasons               [[[1, 1], 0], [[0, 0], 2]]
url                                           {}

## Text: ##
[State and local court rules sometimes make de...]


For reference, please consider:

Joonsuk Park and Claire Cardie. 2018. A Corpus of eRulemaking User Comments
for Measuring Evaluability of Arguments. In Proceedings of the Eleventh
International Conference on Language Resources and Evaluation (LREC 2018),
Miyazaki, Japan. European Language Resources Association (ELRA).

last change
OMS 21.07.2024
"""
from corpus_loader import CorpusLoader
import pandas as pd


class CDCPPropositionPrepper:
    """
    Extracts the proposition labels and corresponding text instances from
    corpus.
    """

    @staticmethod
    def instance_extraction(ann, texts):
        """
        Extracts labels and corresponding text spans from DataFrames.

        Parameters:
            ann, DataFrame: with annotations or propositions as well as
                            proposition offsets
            texts, DataFrame: with texts

        Return:
            DataFrame: with two colums corresponding to propositions and
                       their labels
        """
        labels = []
        instances = []

        for label, span, text in zip(ann["prop_labels"],
                                     ann["prop_offsets"],
                                     texts["text"]):

            for i in range(len(label)):
                labels.append(label[i])

                split_indices = span[i]
                # Text is a list from the df with one single entry
                instance = text[0][split_indices[0]:split_indices[1]]
                instances.append(instance)

        training_data = zip(labels, instances)
        df = pd.DataFrame(training_data, columns=["label", "text"])

        return df


class CDCPRelationPrepper:
    """
    Extracts the relation labels and corresponding text instances from
    corpus.
    """

    def pair_extraction(self, ann, texts, directed=True, window=False,
                        window_size=1):
        """
        Extracts labels and corresponding ordered pairs from each document.

        Parameters:
            ann, DataFrame: with annotations or propositions as well as
                            proposition offsets
            texts, DataFrame: with texts
            directed, bool: if True, only assign edge value to ordered pairs,
                            if False, pairs and their mirrors are asigned a
                            value
            window, bool: if True, only pairs from window will be considered
            window_size, int: size if window for each EDU to the left and
                              right

        Return:
            DataFrame: with relation labels for pair of propositions
        """
        instances = []
        labels = []

        for evidences, reasons, span, text in zip(ann["evidences"],
                                                  ann["reasons"],
                                                  ann["prop_offsets"],
                                                  texts["text"]):
            propositions = []

            # Extract all propositions from given text document
            for i in range(len(span)):
                split_indices = span[i]

                prop = text[0][split_indices[0]:split_indices[1]]
                propositions.append(prop)

            # Consider reasons and evidences together as support of proposition
            support = reasons + evidences

            if window:
                pairs = self.pair_labeling_window(propositions,
                                                  support,
                                                  directed,
                                                  window_size)
            else:
                pairs = self.pair_labeling_wo_replacement(propositions,
                                                          support,
                                                          directed)
            instances += pairs[0]
            labels += pairs[1]

        training_data = zip(labels, instances)
        df = pd.DataFrame(training_data, columns=["label", "pairs"])

        return df

    @staticmethod
    def parse_all_supp_pairs(propositions, support):
        """
        Creates proposition pairs of those that have a support relation
        according to the support annotation.

        Support looks as follows in CDCP:

            [[1, 1], 0] // proposition 1 supports proposition 0
            [[1,1],2] // proposition 1 supports proposition 2
            [[1,3],5] // propositions 1, 2, and 3 collectively support
                          proposition 5

        Parameters:
            propositions, list: with propositions from document
            support, list: with indices, indicating support relations.

        Return:
            set: with proposition tuples where the left proposition supports
                 the right proposition.
        """
        support_pairs = set()

        for relation in support:
            if relation[0][0] == relation[0][1]:  # for example [[1,1],2]
                support_pairs.add((propositions[relation[0][0]], propositions[
                    relation[1]]))
            else:  # for example [[1,3],5]
                diff = relation[0][1] - relation[0][0]
                index = relation[0][0]
                i = 0
                while i <= diff:
                    support_pairs.add((propositions[index + i], propositions[
                        relation[1]]))
                    i += 1

        return support_pairs

    @staticmethod
    def support_status_pairs(propositions, support, directed):
        prop1, prop2, = propositions
        support_pairs = support
        support_status = "None"
        if directed:
            if (prop1, prop2) in support_pairs:
                support_status = "sup"
        else:
            if (prop1, prop2) in support_pairs:
                support_status = "sup"
            elif (prop2, prop1) in support_pairs:
                support_status = "sup"

        return support_status

    def pair_labeling_wo_replacement(self,
                                     propositions,
                                     support,
                                     directed=True):
        """
        From the given propositions and the support, each proposition is
        paired with each other proposition, except for itself. The result
        is a list of ordered pairs sampled without replacement.

        ## Directedness: ##

        Directed: When any of the ordered pairs have a support relation,
        then their label is 1, otherwise 0.
        Undirected: If support exists for one tuple, e.g., (a, b),
        then support will equally be considered for (b, a), and the value for
        both pairs is set as 1.

        For example: permutation of 2 out of 4 should result in 12 pairs.

        4!/(4-2)! = 12

        Parameters:
            propositions, list: with propositions as strings
            support, list: with indices indicating relations between
                           propositions
            directed, bool: True if direction of propositions is considered,
                            that is, order of tuples does matter

        Return:
            tuple: with list of tuples of propositions and corresponding labels
        """
        support_pairs = self.parse_all_supp_pairs(propositions, support)
        prop_pairs = []
        labels = []
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions):
                if i != j:
                    support_status = self.support_status_pairs([prop1, prop2],
                                                               support_pairs,
                                                               directed)

                    prop_pairs.append((prop1, prop2))
                    labels.append(support_status)
        return prop_pairs, labels

    def pair_labeling_window(self,
                             propositions,
                             support,
                             directed,
                             window_size):
        """
        Only pairs within a window for each set of propositions are considered

        Parameters:
            propositions, list: with propositions as strings
            support, list: with indices indicating relations between
                           propositions
            directed, bool: True if direction of propositions is considered,
                            that is, order of tuples does matter
            window_size, int: size if window for each EDU to the left and
                              right

        Return:
            tuple: with list of tuples of propositions and corresponding labels
        """
        assert type(window_size) == int
        support_pairs = self.parse_all_supp_pairs(propositions, support)
        prop_pairs = []
        labels = []

        text_len = len(propositions)

        for i, prop1 in enumerate(propositions):
            start = max(0, i - window_size)
            # +1 to include the element at i + window_size
            end = min(text_len, i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    prop2 = propositions[j]

                    support_status = self.support_status_pairs([prop1, prop2],
                                                           support_pairs,
                                                           directed)
                    prop_pairs.append((prop1, prop2))
                    labels.append(support_status)

        return prop_pairs, labels


def proposition_statistic(training_annotation,
                          training_text,
                          test_annotation,
                          test_text):
    """
    Park & Cardie (2018, p. 1624):

    "We annotated 731 user comments on Consumer Debt Collection Practices (
    CDCP) rule by the Consumer Financial Protection Bureau (CFPB)"

    Park & Cardie (2018, p. 1627):
    Policy: 815 Value: 2182 Fact: 785 Testimony: 1117 Reference: 32
    """
    propositions = CDCPPropositionPrepper()

    training_set = propositions.instance_extraction(training_annotation,
                                                  training_text)
    print(training_set.head())
    print(training_set.tail())

    fact_train = len(training_set[training_set["label"] == "fact"])
    policy_train = len(training_set[training_set["label"] == "policy"])
    reference_train = len(training_set[training_set["label"] == "reference"])
    testimony_train = len(training_set[training_set["label"] == "testimony"])
    value_train = len(training_set[training_set["label"] == "value"])

    #### Test set ####

    test_set = propositions.instance_extraction(test_annotation, test_text)
    print(test_set.head())
    print(test_set.tail())

    fact_test = len(test_set[test_set["label"] == "fact"])
    policy_test = len(test_set[test_set["label"] == "policy"])
    reference_test = len(test_set[test_set["label"] == "reference"])
    testimony_test = len(test_set[test_set["label"] == "testimony"])
    value_test = len(test_set[test_set["label"] == "value"])

    documents_n = len(training_text) + len(test_text)
    edus_n = len(training_set) + len(test_set)

    print(f"""
        Corpus statistics:
         %
        All documents from corpus: {documents_n}
        Training split: {len(training_text)}, {len(training_text) / documents_n} %
        Test split: {len(test_text)}, {len(test_text) / documents_n} %
        Average EDUs per document: {edus_n / documents_n}
        
        Fact training: {fact_train} test: {fact_test}, in sum {fact_train + fact_test}
        Policy training: {policy_train} test: {policy_test}, in sum {policy_train + policy_test}
        Reference training: {reference_train} test: {reference_test}, in sum {reference_train + reference_test}
        Testimony training: {testimony_train} test: {testimony_test}, in sum {testimony_train + testimony_test}
        Value training: {value_train} test: {value_test}, in sum {value_train + value_test}
        """)


def relation_statistics(training_annotation,
                        training_text,
                        test_annotation,
                        test_text):
    """
    Park & Cardie (2018, p. 1624):

    "the resulting dataset contains 4931 elementary unit and 1221 support
    relation annotations"

    However, the DFKI Speech, Language, and Technology unit reports on
    HuggingFace a total of 1426 support relations for train and test sets.
    (https://huggingface.co/datasets/DFKI-SLT/cdcp/blob/main/README.md)
    """
    relations = CDCPRelationPrepper()
    directed = True  # Consider order of relation pairs
    window = True  # If True only pairs within a window are considered
    window_size = 3  # Size of window to the left and right

    training_set = relations.pair_extraction(training_annotation,
                                             training_text,
                                             directed,
                                             window,
                                             window_size)

    test_set = relations.pair_extraction(test_annotation,
                                         test_text,
                                         directed,
                                         window,
                                         window_size)
    print(training_set.head(5))
    print(training_set.shape)
    print(test_set.head(5))
    print(test_set.shape)

    all_relations_train = len(training_set)
    all_relations_test = len(test_set)

    train_n = len(training_set[training_set["label"] == "sup"])
    test_n = len(test_set[test_set["label"] == "sup"])
    relation_n = train_n + test_n

    print(f"""
          Relation statistics:
          
          All relations in training: {all_relations_train}
          Support relations in training set: {train_n}
          
          All relation in test: {all_relations_test}
          Support relations in test set: {test_n}
          
          Support relations in sum: {relation_n}
          """)


def test_relation_prepper():
    relations = CDCPRelationPrepper()
    example_doc = ['Allow the States and/or local courts administer lawsuits.', ' Regardless of any added rules, documents, etc. none will help increase debtors attending hearings.', ' Since 1964 I have seen no significant increase of consumers attending hearings.', ' I have, however, seen the major reason being attributed to not understanding due process by consumers.']
    example_ann = [[[1, 1], 0], [[2, 3], 0]]
    pairs = relations.pair_labeling_wo_replacement(example_doc, example_ann)

    assert len(relations.parse_all_supp_pairs(example_doc, example_ann)) == 3
    # Permutation of 2 out of 4 should result in 12 pairs
    assert len(pairs[0]) == 12


if __name__ == "__main__":
    url_cdcp = "https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip"
    cdcp_save_path = "../data/corpus_1/"
    cdcp_path_train = "../data/corpus_1/cdcp/train/"
    cdcp_path_test = "../data/corpus_1/cdcp/test/"
    cdcp_file_type_ann = ".ann.json"
    cdcp_file_type_text = ".txt"

    corpus_loader = CorpusLoader()
    corpus_loader.download_file_from(url_cdcp, cdcp_save_path)

    training_annotation = corpus_loader.load_files_from(cdcp_path_train,
                                                        cdcp_file_type_ann)
    training_text = corpus_loader.load_files_from(cdcp_path_train,
                                                  cdcp_file_type_text)

    test_annotation = corpus_loader.load_files_from(cdcp_path_test,
                                                    cdcp_file_type_ann)
    test_text = corpus_loader.load_files_from(cdcp_path_test,
                                              cdcp_file_type_text)

    proposition_statistic(training_annotation,
                          training_text,
                          test_annotation,
                          test_text)

    relation_statistics(training_annotation,
                        training_text,
                        test_annotation,
                        test_text)

    test_relation_prepper()
