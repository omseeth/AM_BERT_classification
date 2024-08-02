"""
This script helps with downloading and preparing data from a URL.
Specifically two corpora can be downloaded:

1)Cornell eRulemaking Corpus – CDCP, available here:
https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip.

2) Argumentative Microtext Corpus Part One, available here:
https://github.com/peldszus/arg-microtexts/archive/refs/heads/master.zip

3) Argumentative Microtext Corpus Part Two, available here:
https://github.com/discourse-lab/arg-microtexts-part2/archive/refs/heads/master.zip


For reference, please consider:

(1) Joonsuk Park and Claire Cardie. 2018. A Corpus of eRulemaking User Comments
for Measuring Evaluability of Arguments. In Proceedings of the Eleventh
International Conference on Language Resources and Evaluation (LREC 2018),
Miyazaki, Japan. European Language Resources Association (ELRA).

(2) Andreas Peldszus and Manfred Stede. An annotated corpus of argumentative
microtexts. In D. Mohammed, and M. Lewinski, editors, Argumentation and
Reasoned Action - Proc. of the 1st European Conference on Argumentation,
Lisbon, 2015. College Publications, London, 2016

(3) Maria Skeppstedt, Andreas Peldszus and Manfred Stede. More or less
controlled elicitation of argumentative text: Enlarging a microtext corpus via
crowdsourcing. In Proc. 5th Workshop in Argumentation Mining (at EMNLP),
Brussels, 2018

last change
OMS 21.07.2024
"""
import zipfile
from io import BytesIO
import os
import json

import requests
from tqdm import tqdm
import pandas as pd
import xml.etree.ElementTree as ET


class CorpusLoader:
    """
    This class provides methods to download and extract files from a zip file.
    """

    @staticmethod
    def download_file_from(url_to_zip, path):
        """
        Downloading zip file from given URL.

        Parameter:
            url_to_zip, str:

        Return:
            None: If path to data already exists, method terminates,
                  otherwise content from url is downloaded and saved at
                  [path]
        """
        # Check if the target extraction path exists
        if os.path.exists(path):
            print(f"Path {path} already exists. Skipping download.")
            return

        response = requests.get(url_to_zip)
        response.raise_for_status()  # Check that the request was successful

        total_size = int(response.headers.get("content-length", 0))

        # Code generated with GPT4o from OpenAI 09.07.2024
        with tqdm(total=total_size,
                  unit="B",
                  unit_scale=True,
                  desc="Downloading the data from URL") as pbar:

            file_bytes = BytesIO()

            # Code generated with GPT4o from OpenAI 09.07.2024
            for chunk in response.iter_content(chunk_size=8192):
                file_bytes.write(chunk)
                pbar.update(len(chunk))

        # Code generated with GPT4o from OpenAI 09.07.2024
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(path)

    @staticmethod
    def extract_xml_tree(root):
        """
        Helps with extracting the annotations from a given .xml file of the
        Microtext corpus.

        For example:

        <?xml version='1.0' encoding='UTF-8'?>
        <arggraph id="micro_XXX" topic_id="Test" stance="pro">
            <edu id="e1"><![CDATA[Skiing is not good for the
            environment]]></edu>
            <edu id="e2"><![CDATA[because it causes pollution.]]></edu>
            <adu id="a1" type="contra"/>
            <adu id="a2" type="contra"/>
            <edge id="c7" src="e1" trg="a1" type="seg"/>
            <edge id="c7" src="e2" trg="a2" type="seg"/>
            <edge id="c1" src="a2" trg="a1" type="sup"/>
        </arggraph>

        Parameter:
            root, xml.etree: with edus and adus and edges

        Return:
            ann_src, dict: with entries for edus, adus, and edges
        """
        ann_src = {"edus": {}, "adus": {}, "edges": []}

        edus = root.findall("edu")
        for edu in edus:
            edu_id = edu.get("id")
            edu_text = edu.text.strip()
            ann_src["edus"].update({edu_id: edu_text})

        adus = root.findall("adu")
        for adu in adus:
            adu_id = adu.get("id")
            adu_type = adu.get("type")
            ann_src["adus"].update({adu_id: adu_type})

        edges = root.findall('edge')
        for edge in edges:
            edge_id = edge.get("id")
            src = edge.get("src")
            tgt = edge.get("trg")
            edge_type = edge.get("type")
            ann_src["edges"].append([edge_id, src, tgt, edge_type])

        return ann_src

    def file_generator(self, data_path, file_type):
        """
        This generator yields all files ending with provided file type (e.g.
        ".ann.json")

        Parameter:
            data_path, str: to the files
        """
        files = os.listdir(data_path)
        sorted_files = sorted(files)

        for filename in sorted_files:
            if filename.endswith(file_type):
                file_path = os.path.join(data_path, filename)

                if filename.endswith(".xml"):
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    item = self.extract_xml_tree(root)
                    yield item
                else:
                    with open(file_path, "r") as file:
                        if filename.endswith(".json"):
                            item = json.load(file)
                        if filename.endswith(".txt"):
                            item = file.readlines()
                            item = {"text": item}
                        yield item

    def load_files_from(self, data_path, file_type):
        """
        Helper method to convert generator object into pandas dataframe.

        Parameter:
            folder, str: should be either "train" or "test" for the CDCP corpus

        Return:
            DataFrame: with entries from json files
        """

        data_gen = self.file_generator(data_path, file_type)
        df = pd.DataFrame(data_gen)

        return df


if __name__ == "__main__":

    url_cdcp = "https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip"
    cdcp_save_path = "../data/corpus_1/"
    cdcp_path_train = "../data/corpus_1/cdcp/train/"
    cdcp_file_type_ann = ".ann.json"
    cdcp_file_type_text = ".txt"

    url_microtext = "https://github.com/discourse-lab/arg-microtexts-part2/archive/refs/heads/master.zip"
    microtext_save_path = "../data/corpus_3/"
    microtext_path = "../data/corpus_3/arg-microtexts-part2-master/corpus/"
    microtext_file_type_ann = ".xml"
    microtext_file_type_text = ".txt"

    corpus_loader = CorpusLoader()

    corpus_loader.download_file_from(url_cdcp, cdcp_save_path)
    cdcp_annotation = corpus_loader.load_files_from(cdcp_path_train,
                                                    cdcp_file_type_ann)
    cdcp_text = corpus_loader.load_files_from(cdcp_path_train,
                                              cdcp_file_type_text)
    
    print(cdcp_annotation.iloc[0])
    print(cdcp_annotation.head(59)["reasons"])
    print(cdcp_text.head())

    corpus_loader.download_file_from(url_microtext, microtext_save_path)
    microtext_annotation = corpus_loader.load_files_from(microtext_path,
                                                         microtext_file_type_ann)
    microtext_text = corpus_loader.load_files_from(microtext_path,
                                                   microtext_file_type_text)

    print(microtext_annotation.iloc[0])
    print(microtext_annotation.head())
    print(microtext_text.head())
