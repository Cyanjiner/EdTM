from lxml import etree
from collections import defaultdict
import pandas as pd
import html

__author__ = "Jon Ball"
__version__ = "Autumn 2022"

# Thanks to Stefan Behnel, creator of lxml (https://lxml.de/index.html)

# Namespaces for ERIC records
nsmap = {"eric": "http://www.eric.ed.gov",
         "dc": "http://purl.org/dc/elements/1.1/",
         "dcterms": "http://purl.org/dc/terms/"}

class ERICparser:
    """
    A class for parsing .xml files downloaded from the ERIC database.
    Intended to parse one and only one .xml file at a time, returning relevant metadata.
    https://eric.ed.gov/?download
    """
    def __init__(self, namespaces=nsmap):
        self.nsmap = namespaces
        self.tree = None
        self.root = None
        self.num_records = 0

    def parse(self, path_to_xml):
        """
        Call lxml.etree.parse() and save the .xml tree to self.
        """
        self.tree = etree.parse(path_to_xml) # Parse the .xml file
        self.root = self.tree.getroot()
        self.num_records = len(self.root.getchildren()) # Save the number of records in the .xml file

    def iter_metadata(self):
        """
        Function called internally to produce metadata for item records in an ERIC .xml file.
        """
        for child in self.root.iterchildren(): # Iteratively yield the metadata for each record
            yield child.xpath("metadata", namespaces=self.nsmap)[0]

    def iter_field(self, element:str):
        """
        Function called by users to iteratively yield a single metadata field from each record.
        """
        for rec in self.iter_metadata():
            rectext = rec.xpath("dc:" + element, namespaces=self.nsmap)[0].text
            if rectext:
                yield html.unescape(rectext).strip()

    def data_fields(self, elements:list, flatten=False):
        """
        Function called by users to pull lists of record elements of their choosing.

        Args:
            elements: List of ERIC data fields to pull from .xml file.

        Returns:
            dict mapping user's provided element to lists of lists of elements
        """
        d = defaultdict(list)
        for elem in elements: # For each element provided by user
            for rec in self.iter_metadata(): # For each record's metadata
                rlist = rec.xpath("dc:" + elem, namespaces=self.nsmap) # Pull data field from record
                rlist = [
                    html.unescape(f.text).strip() for f in rlist if f.text is not None
                    ] # Create list of non-null data fields
                if rlist:
                    d[elem].append(rlist)
                else:
                    d[elem].append(None) # If the record lacks specfied fields, append None
        if flatten: # Flatten the lists of lists. Record order is lost, but iteration is easier
            d = {k: [elem for rec in v if rec for elem in rec] for k, v in d.items()} 
        else:
            d = dict(d) # defaultdict to dict
        
        return d


class APIparser:
    """
    A class for parsing .xml files queried using the ERIC API.
    Intended to parse one and only one .xml file at a time, returning relevant metadata.
    https://eric.ed.gov/?api
    """
    def __init__(self):
        self.tree = None
        self.root = None
        self.fields = None
        self.num_docs = 0

    def parse(self, path_to_xml):
        """
        Call lxml.etree.parse() and save the .xml tree to self.
        """
        self.tree = etree.parse(path_to_xml) # Parse the .xml file
        self.root = self.tree.getroot()
        # Save the number of docs in the response .xml file
        self.num_docs = len(
            self.root.xpath("result")[0].getchildren())
        self.fields = set(
            [str(s) for s in self.root.xpath("//@name")])

    def iter_docs(self):
        """
        Function called internally to produce metadata for doc records in an ERIC API response .xml.
        """
        for child in self.root.xpath("result")[0].iterchildren(): 
            # Iteratively yield the metadata for each doc
            yield child

    def data_fields(self, return_df=False):
        """
        Function called by users to render an API response either as dict or pd.DataFrame.

        Args:
            return_df: optional flag for returning a pandas.DataFrame

        Returns:
            dict mapping fields to list of strings (or int for year); or pandas.DataFrame
        """
        d = defaultdict(list)
        for doc in self.iter_docs(): # For each doc's metadata
            # Declare an empty set which will be used to record null values
            field_check = set()
            # Store year of publication
            for i in doc.xpath("int"):
                name = str(i.xpath("@name")[0])
                d[name].append(int(i.text))
                field_check.add(name)
            # Store single string elements
            for s in doc.xpath("str"):
                name = str(s.xpath("@name")[0])
                d[str(name)].append(s.text)
                field_check.add(name)
            # Store arrayed elements as strings
            for arr in doc.xpath("arr"):
                name = str(arr.xpath("@name")[0])
                if len(arr.xpath("str")) == 1:
                    d[name].append(arr.xpath("str")[0].text)
                    field_check.add(name)
                if len(arr.xpath("str")) > 1:
                    d[name].append("; ".join([s.text for s in arr.xpath("str")]))
                    field_check.add(name)
            # Make sure all values in d are of uniform length
            for field in self.fields.difference(field_check):
                d[field].append(None)

        if return_df: # Return the API results as a pandas DataFrame
            return pd.DataFrame(d).drop(columns=["response"])
        else:
            return dict(d)