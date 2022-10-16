from lxml import etree
from collections import defaultdict
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
    A class for parsing .xml files downloaded from the ERIC database or ERIC API.
    Intended to parse one and only one .xml file at a time, returning relevant metadata.
    https://eric.ed.gov/?download , https://eric.ed.gov/?api
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